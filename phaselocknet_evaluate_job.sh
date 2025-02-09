#!/bin/bash
#
#SBATCH --job-name=phaselocknet_evaluate
#SBATCH --out="slurm-%A_%a.out"
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1 --exclude=node[017-094,097,098],dgx001,dgx002
#SBATCH --array=0-29
#SBATCH --partition=normal --time=2-0
#SBATCH --requeue

offset=0
job_idx=$(($SLURM_ARRAY_TASK_ID + $offset))
batch_size=32

declare -a list_dir_model=(
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch01"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch02"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch03"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch04"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch05"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch06"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch07"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch08"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch09"
    "models/sound_localization/simplified_IHC3000_delayed_integration/arch10"
    "models/sound_localization/simplified_IHC3000/arch01"
    "models/sound_localization/simplified_IHC3000/arch02"
    "models/sound_localization/simplified_IHC3000/arch03"
    "models/sound_localization/simplified_IHC3000/arch04"
    "models/sound_localization/simplified_IHC3000/arch05"
    "models/sound_localization/simplified_IHC3000/arch06"
    "models/sound_localization/simplified_IHC3000/arch07"
    "models/sound_localization/simplified_IHC3000/arch08"
    "models/sound_localization/simplified_IHC3000/arch09"
    "models/sound_localization/simplified_IHC3000/arch10"

    "models/spkr_word_recognition/simplified_IHC3000/arch0_0000"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0001"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0002"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0004"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0006"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0007"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0008"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0009"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0016"
    "models/spkr_word_recognition/simplified_IHC3000/arch0_0017"
)
dir_model="${list_dir_model[$job_idx]}"
echo $HOSTNAME $job_idx $dir_model

module add openmind8/anaconda
source activate dtu

if [[ $dir_model == *"sound_localization"* ]]
then
    declare -a list_tag_expt=(
        "itd_ild_weighting"
        "itd_threshold"
        "v01_eval_mit_bldg46room1004_tenoise"
        "speech_in_noise_in_reverb_v04"
        "minimum_audible_angle_interpolated"
        "mp_spectral_cues"
        "new_ears"
        "spectral_smoothing"
        "precedence_effect_localization"
        "bandwidth_dependency"
    )
    for tag_expt in "${list_tag_expt[@]}"
    do
        regex_eval="/om2/user/msaddler/phaselocknet/stimuli/sound_localization/evaluation/$tag_expt/stim*.hdf5"
        fn_eval_output="eval_phaselocknet_localization_$tag_expt.csv"
        echo "|__ tag_expt=$tag_expt"
        echo "|__ regex_eval=$regex_eval"
        echo "|__ fn_eval_output=$fn_eval_output"
        python -u phaselocknet_evaluate.py \
        -m "$dir_model" \
        -e "$regex_eval" \
        -fe "$fn_eval_output" \
        -b $batch_size \
        -n 8 \
        -wp 1
    done

    regex_eval="/om2/user/msaddler/phaselocknet/stimuli/sound_localization/optimization/valid/stim*.hdf5"
    fn_eval_output="eval_phaselocknet_localization_valid.csv"
    echo "|__ tag_expt=$tag_expt"
    echo "|__ regex_eval=$regex_eval"
    echo "|__ fn_eval_output=$fn_eval_output"
    python -u phaselocknet_evaluate.py \
    -m "$dir_model" \
    -e "$regex_eval" \
    -fe "$fn_eval_output" \
    -b $batch_size \
    -n 8 \
    -wp 0
else
    declare -a list_tag_expt=(
        "human_experiment_v00_foreground60dbspl"
        "human_experiment_v00_inharmonic_foreground60dbspl"
        "speech_in_synthetic_textures"
        "pitch_altered_v00"
        "hopkins_moore_2009"
    )
    for tag_expt in "${list_tag_expt[@]}"
    do
        regex_eval="/om2/user/msaddler/phaselocknet/stimuli/spkr_word_recognition/evaluation/$tag_expt/stim*.hdf5"
        fn_eval_output="eval_phaselocknet_spkr_word_recognition_$tag_expt.csv"
        echo "|__ tag_expt=$tag_expt"
        echo "|__ regex_eval=$regex_eval"
        echo "|__ fn_eval_output=$fn_eval_output"
        python -u phaselocknet_evaluate.py \
        -m "$dir_model" \
        -e "$regex_eval" \
        -fe "$fn_eval_output" \
        -b $batch_size \
        -n 8 \
        -wp 0
    done
fi
