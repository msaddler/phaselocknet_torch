import numpy as np
import torch
import torchaudio

import util_torch


class PeripheralModel(torch.nn.Module):
    def __init__(
        self,
        sr_input=40e3,
        sr_output=20e3,
        min_cf=1e2,
        max_cf=1e4,
        num_cf=60,
        fir_dur=None,
        scale_gammatone_filterbank=8.0,
        ihc_lowpass_cutoff=4.8e3,
        anf_per_channel=200,
    ):
        """
        PyTorch port of auditory nerve model from Heinz et al. (2001):
        https://github.com/ModelDBRepository/37436/blob/master/anmodheinz00.c
        """
        super().__init__()
        self.sr_input = sr_input
        self.sr_output = sr_output
        if self.sr_output is None:
            self.sr_output = self.sr_input
        self.cfs = self.x2f(np.linspace(self.f2x(min_cf), self.f2x(max_cf), num_cf))
        self.gammatone_filterbank = util_torch.GammatoneFilterbank(
            sr=self.sr_input,
            fir_dur=fir_dur,
            cfs=self.cfs,
            dtype=torch.float32,
        )
        self.scale_gammatone_filterbank = scale_gammatone_filterbank
        self.ihc_nonlinearity = IHCNonlinearity()
        self.ihc_lowpassfilter = IHCLowpassFilter(
            sr_input=self.sr_input,
            sr_output=self.sr_output,
            cutoff=ihc_lowpass_cutoff,
        )
        self.neural_adaptation = NeuralAdaptation(sr=self.sr_output)
        self.anf_per_channel = anf_per_channel

    def f2x(self, f):
        """
        Map characteristic frequency to basilar membrane position
        """
        msg = "frequency out of human range"
        assert np.all(np.logical_and(f >= 20, f <= 20677)), msg
        x = (1.0 / 0.06) * np.log10((f / 165.4) + 0.88)
        return x

    def x2f(self, x):
        """
        Map basilar membrane position to characteristic frequency
        """
        msg = "basilar membrane distance out of human range"
        assert np.all(np.logical_and(x >= 0, x <= 35)), msg
        f = 165.4 * (np.power(10.0, (0.06 * x)) - 0.88)
        return f

    def spike_generator(self, rate):
        """
        Map auditory nerve firing rates to spike counts
        (independent binomial sampling at each time point)
        """
        if self.anf_per_channel is None:
            return rate
        p = rate / self.sr_output
        spikes = (
            torch.rand(
                size=(self.anf_per_channel, *p.shape),
                device=rate.device,
            )
            < p[None, :]
        )
        return spikes.sum(dim=0).to(rate.dtype)

    def forward(self, x):
        """
        Map sound waveforms to auditory nerve spike counts

        Args
        ----
        x (torch.tensor): sound waveforms with shape
            [batch, time, 2] or [batch, channels, time]

        Returns
        -------
        x (torch.tensor): auditory nerve spike counts with shape
            [batch, channels, characteristic frequency, time]
        """
        if x.shape[-1] == 2:
            x = torch.swapaxes(x, -1, -2)
        x = self.gammatone_filterbank(x)
        if self.scale_gammatone_filterbank is not None:
            x = self.scale_gammatone_filterbank * x
        x = self.ihc_nonlinearity(x)
        x = self.ihc_lowpassfilter(x)
        x = self.neural_adaptation(x)
        x = self.spike_generator(x)
        return x


class IHCNonlinearity(torch.nn.Module):
    def __init__(
        self,
        ihc_asym=3,
        ihc_k=1225.0,
    ):
        """ """
        super().__init__()
        self.ihc_beta = torch.tensor(
            np.tan(np.pi * (-0.5 + 1 / (ihc_asym + 1))),
            dtype=torch.float32,
        )
        self.ihc_k = torch.tensor(ihc_k, dtype=torch.float32)

    def forward(self, x):
        """ """
        x = torch.atan(self.ihc_k * x + self.ihc_beta) - torch.atan(self.ihc_beta)
        x = x / (np.pi / 2 - torch.atan(self.ihc_beta))
        return x


class IHCLowpassFilter(torch.nn.Module):
    def __init__(
        self,
        sr_input=40e3,
        sr_output=20e3,
        cutoff=4.8e3,
        n=7,
    ):
        """ """
        super().__init__()
        self.sr_input = sr_input
        self.sr_output = sr_output
        self.n = n
        c = 2 * self.sr_input
        c1LPihc = (c - 2 * np.pi * cutoff) / (c + 2 * np.pi * cutoff)
        c2LPihc = 2 * np.pi * cutoff / (2 * np.pi * cutoff + c)
        b = torch.tensor([c2LPihc, c2LPihc], dtype=torch.float32)
        a = torch.tensor([1.0, -c1LPihc], dtype=torch.float32)
        self.register_buffer("b", b)
        self.register_buffer("a", a)
        self.stride = int(sr_input / sr_output)
        msg = f"{sr_input=} and {sr_output=} require non-integer stride"
        assert np.isclose(self.stride, sr_input / sr_output), msg

    def forward(self, x):
        """ """
        for _ in range(self.n):
            x = torchaudio.functional.lfilter(
                waveform=x,
                a_coeffs=self.a,
                b_coeffs=self.b,
            )
        if not self.sr_output == self.sr_input:
            x = x[..., :: self.stride]
        return x


class NeuralAdaptation(torch.nn.Module):
    def __init__(
        self,
        sr=20e3,
        VI=5e-4,
        VL=5e-3,
        PG=3e-2,
        PL=6e-2,
        PIrest=1.2e-2,
        PImax=6e-1,
        spont=5e1,
    ):
        """ """
        super().__init__()
        self.sr = torch.tensor(sr, dtype=torch.float32)
        self.VI = torch.tensor(VI, dtype=torch.float32)
        self.VL = torch.tensor(VL, dtype=torch.float32)
        self.PG = torch.tensor(PG, dtype=torch.float32)
        self.PL = torch.tensor(PL, dtype=torch.float32)
        self.PIrest = torch.tensor(PIrest, dtype=torch.float32)
        self.PImax = torch.tensor(PImax, dtype=torch.float32)
        self.spont = torch.tensor(spont, dtype=torch.float32)
        self.ln2 = torch.log(torch.tensor(2.0))

    def forward(self, ihcl):
        """ """
        SPER = 1 / self.sr
        CI = torch.ones_like(ihcl[..., 0]) * self.spont / self.PIrest
        CL = CI * (self.PIrest + self.PL) / self.PL
        CG = CL * (1 + self.PL / self.PG) - CI * self.PL / self.PG
        p1 = torch.log(torch.exp(self.ln2 * self.PImax / self.PIrest) - 1)
        p3 = p1 * self.PIrest / self.ln2
        PPI = p3 / p1 * torch.log(1 + torch.exp(p1 * ihcl))
        ifr = torch.ones_like(ihcl) * self.spont
        for k in range(1, ihcl.shape[-1]):
            CI = CI + (SPER / self.VI) * (-PPI[..., k] * CI + self.PL * (CL - CI))
            CL = CL + (SPER / self.VL) * (-self.PL * (CL - CI) + self.PG * (CG - CL))
            ifr[..., k] = CI * PPI[..., k]
        return ifr
