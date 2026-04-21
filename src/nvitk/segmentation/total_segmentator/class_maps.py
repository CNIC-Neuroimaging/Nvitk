"""
TotalSegmentator task class maps.

This module mirrors the upstream TotalSegmentator v2 class taxonomy, as
captured in the legacy :code:`BioImaging/src/values.py`. Every numeric label
ID matches the order used by TotalSegmentator's multi-label outputs so that
downstream mask extraction can look up by integer directly (no extra
name→id lookup table is needed).

Contents
--------
- :data:`CLASS_MAPS` — task-name → ``{id: class_name}`` dict.
- :data:`AVAILABLE_TASKS` — tuple of every task key in :data:`CLASS_MAPS`.
- :data:`CLASS_MAP_5_PARTS` — TotalSegmentator CT "5 parts" grouping.
- :data:`CLASS_MAP_PARTS_MR` — TotalSegmentator MR parts grouping.
- :data:`CLASS_MAP_PARTS_HEADNECK_MUSCLES` — head/neck muscle parts grouping.
- :data:`MAP_TASKID_TO_PARTNAME_CT` / ``_MR`` / ``_HEADNECK_MUSCLES`` —
  nnU-Net task-id → part-name lookup tables (upstream project metadata).
- :data:`COMMERCIAL_MODELS` — upstream commercial model ids; metadata only.

The helper accessors :func:`get_class_map`, :func:`get_class_id` and
:func:`get_class_name` have not changed in shape; they dispatch through
:data:`CLASS_MAPS`.
"""

from __future__ import annotations

from typing import Mapping


_TOTAL_V1: Mapping[int, str] = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "aorta", 8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein", 10: "pancreas",
    11: "adrenal_gland_right", 12: "adrenal_gland_left",
    13: "lung_upper_lobe_left", 14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right", 16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5", 19: "vertebrae_L4", 20: "vertebrae_L3",
    21: "vertebrae_L2", 22: "vertebrae_L1",
    23: "vertebrae_T12", 24: "vertebrae_T11", 25: "vertebrae_T10",
    26: "vertebrae_T9", 27: "vertebrae_T8", 28: "vertebrae_T7",
    29: "vertebrae_T6", 30: "vertebrae_T5", 31: "vertebrae_T4",
    32: "vertebrae_T3", 33: "vertebrae_T2", 34: "vertebrae_T1",
    35: "vertebrae_C7", 36: "vertebrae_C6", 37: "vertebrae_C5",
    38: "vertebrae_C4", 39: "vertebrae_C3", 40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus", 43: "trachea",
    44: "heart_myocardium", 45: "heart_atrium_left",
    46: "heart_ventricle_left", 47: "heart_atrium_right",
    48: "heart_ventricle_right",
    49: "pulmonary_artery", 50: "brain",
    51: "iliac_artery_left", 52: "iliac_artery_right",
    53: "iliac_vena_left", 54: "iliac_vena_right",
    55: "small_bowel", 56: "duodenum", 57: "colon",
    58: "rib_left_1", 59: "rib_left_2", 60: "rib_left_3",
    61: "rib_left_4", 62: "rib_left_5", 63: "rib_left_6",
    64: "rib_left_7", 65: "rib_left_8", 66: "rib_left_9",
    67: "rib_left_10", 68: "rib_left_11", 69: "rib_left_12",
    70: "rib_right_1", 71: "rib_right_2", 72: "rib_right_3",
    73: "rib_right_4", 74: "rib_right_5", 75: "rib_right_6",
    76: "rib_right_7", 77: "rib_right_8", 78: "rib_right_9",
    79: "rib_right_10", 80: "rib_right_11", 81: "rib_right_12",
    82: "humerus_left", 83: "humerus_right",
    84: "scapula_left", 85: "scapula_right",
    86: "clavicula_left", 87: "clavicula_right",
    88: "femur_left", 89: "femur_right",
    90: "hip_left", 91: "hip_right",
    92: "sacrum", 93: "face",
    94: "gluteus_maximus_left", 95: "gluteus_maximus_right",
    96: "gluteus_medius_left", 97: "gluteus_medius_right",
    98: "gluteus_minimus_left", 99: "gluteus_minimus_right",
    100: "autochthon_left", 101: "autochthon_right",
    102: "iliopsoas_left", 103: "iliopsoas_right",
    104: "urinary_bladder",
}


_TOTAL: Mapping[int, str] = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas",
    8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland",
    18: "small_bowel", 19: "duodenum", 20: "colon",
    21: "urinary_bladder", 22: "prostate",
    23: "kidney_cyst_left", 24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4",
    29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1",
    32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10",
    35: "vertebrae_T9", 36: "vertebrae_T8", 37: "vertebrae_T7",
    38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4",
    41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1",
    44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5",
    47: "vertebrae_C4", 48: "vertebrae_C3", 49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart", 52: "aorta", 53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right", 56: "subclavian_artery_left",
    57: "common_carotid_artery_right", 58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left", 60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava", 63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left", 66: "iliac_artery_right",
    67: "iliac_vena_left", 68: "iliac_vena_right",
    69: "humerus_left", 70: "humerus_right",
    71: "scapula_left", 72: "scapula_right",
    73: "clavicula_left", 74: "clavicula_right",
    75: "femur_left", 76: "femur_right",
    77: "hip_left", 78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
    82: "gluteus_medius_left", 83: "gluteus_medius_right",
    84: "gluteus_minimus_left", 85: "gluteus_minimus_right",
    86: "autochthon_left", 87: "autochthon_right",
    88: "iliopsoas_left", 89: "iliopsoas_right",
    90: "brain", 91: "skull",
    92: "rib_left_1", 93: "rib_left_2", 94: "rib_left_3",
    95: "rib_left_4", 96: "rib_left_5", 97: "rib_left_6",
    98: "rib_left_7", 99: "rib_left_8", 100: "rib_left_9",
    101: "rib_left_10", 102: "rib_left_11", 103: "rib_left_12",
    104: "rib_right_1", 105: "rib_right_2", 106: "rib_right_3",
    107: "rib_right_4", 108: "rib_right_5", 109: "rib_right_6",
    110: "rib_right_7", 111: "rib_right_8", 112: "rib_right_9",
    113: "rib_right_10", 114: "rib_right_11", 115: "rib_right_12",
    116: "sternum", 117: "costal_cartilages",
}


_TOTAL_MR: Mapping[int, str] = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas",
    8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_left", 11: "lung_right",
    12: "esophagus",
    13: "small_bowel", 14: "duodenum", 15: "colon",
    16: "urinary_bladder", 17: "prostate",
    18: "sacrum", 19: "vertebrae", 20: "intervertebral_discs",
    21: "spinal_cord",
    22: "heart", 23: "aorta",
    24: "inferior_vena_cava",
    25: "portal_vein_and_splenic_vein",
    26: "iliac_artery_left", 27: "iliac_artery_right",
    28: "iliac_vena_left", 29: "iliac_vena_right",
    30: "humerus_left", 31: "humerus_right",
    32: "scapula_left", 33: "scapula_right",
    34: "clavicula_left", 35: "clavicula_right",
    36: "femur_left", 37: "femur_right",
    38: "hip_left", 39: "hip_right",
    40: "gluteus_maximus_left", 41: "gluteus_maximus_right",
    42: "gluteus_medius_left", 43: "gluteus_medius_right",
    44: "gluteus_minimus_left", 45: "gluteus_minimus_right",
    46: "autochthon_left", 47: "autochthon_right",
    48: "iliopsoas_left", 49: "iliopsoas_right",
    50: "brain",
}


_TOTAL_HIGHRES_TEST: Mapping[int, str] = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas",
    8: "adrenal_gland_right", 9: "adrenal_gland_left",
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland",
    18: "small_bowel", 19: "duodenum", 20: "colon",
    21: "urinary_bladder", 22: "prostate",
    23: "kidney_cyst_left", 24: "kidney_cyst_right",
}


_LUNG_VESSELS: Mapping[int, str] = {1: "lung_vessels", 2: "lung_trachea_bronchia"}


_OCULOMOTOR_MUSCLES: Mapping[int, str] = {
    1: "skull",  # auxiliary class
    2: "eyeball_right", 3: "lateral_rectus_muscle_right",
    4: "superior_oblique_muscle_right", 5: "levator_palpebrae_superioris_right",
    6: "superior_rectus_muscle_right", 7: "medial_rectus_muscle_left",
    8: "inferior_oblique_muscle_right", 9: "inferior_rectus_muscle_right",
    10: "optic_nerve_left",
    11: "eyeball_left", 12: "lateral_rectus_muscle_left",
    13: "superior_oblique_muscle_left", 14: "levator_palpebrae_superioris_left",
    15: "superior_rectus_muscle_left", 16: "medial_rectus_muscle_right",
    17: "inferior_oblique_muscle_left", 18: "inferior_rectus_muscle_left",
    19: "optic_nerve_right",
}


_COVID: Mapping[int, str] = {1: "lung_covid_infiltrate"}


_CEREBRAL_BLEED: Mapping[int, str] = {1: "intracerebral_hemorrhage"}


_HIP_IMPLANT: Mapping[int, str] = {1: "hip_implant"}


_CORONARY_ARTERIES: Mapping[int, str] = {1: "coronary_arteries"}


_BODY: Mapping[int, str] = {1: "body_trunc", 2: "body_extremities"}


_VERTEBRAE_MR: Mapping[int, str] = {
    1: "sacrum",
    2: "vertebrae_L5", 3: "vertebrae_L4", 4: "vertebrae_L3",
    5: "vertebrae_L2", 6: "vertebrae_L1",
    7: "vertebrae_T12", 8: "vertebrae_T11", 9: "vertebrae_T10",
    10: "vertebrae_T9", 11: "vertebrae_T8", 12: "vertebrae_T7",
    13: "vertebrae_T6", 14: "vertebrae_T5", 15: "vertebrae_T4",
    16: "vertebrae_T3", 17: "vertebrae_T2", 18: "vertebrae_T1",
    19: "vertebrae_C7", 20: "vertebrae_C6", 21: "vertebrae_C5",
    22: "vertebrae_C4", 23: "vertebrae_C3", 24: "vertebrae_C2",
    25: "vertebrae_C1",
}


_PLEURAL_PERICARD_EFFUSION: Mapping[int, str] = {
    # 1: "lung_pleural" (reserved / unused upstream)
    2: "pleural_effusion",
    3: "pericardial_effusion",
}


_LIVER_VESSELS: Mapping[int, str] = {1: "liver_vessels", 2: "liver_tumor"}


_VERTEBRAE_BODY: Mapping[int, str] = {1: "vertebrae_body", 2: "intervertebral_discs"}


_HEARTCHAMBERS_HIGHRES: Mapping[int, str] = {
    1: "heart_myocardium",
    2: "heart_atrium_left",
    3: "heart_ventricle_left",
    4: "heart_atrium_right",
    5: "heart_ventricle_right",
    6: "aorta",
    7: "pulmonary_artery",
}


_APPENDICULAR_BONES: Mapping[int, str] = {
    1: "patella", 2: "tibia", 3: "fibula",
    4: "tarsal", 5: "metatarsal", 6: "phalanges_feet",
    7: "ulna", 8: "radius", 9: "carpal",
    10: "metacarpal", 11: "phalanges_hand",
}


_APPENDICULAR_BONES_MR: Mapping[int, str] = {
    1: "patella", 2: "tibia", 3: "fibula",
    4: "tarsal", 5: "metatarsal", 6: "phalanges_feet",
    7: "ulna", 8: "radius",
}


_APPENDICULAR_BONES_AUXILIARY: Mapping[int, str] = {
    12: "humerus", 13: "femur", 14: "liver", 15: "spleen",
}


_TISSUE_TYPES: Mapping[int, str] = {
    1: "subcutaneous_fat", 2: "torso_fat", 3: "skeletal_muscle",
}


_TISSUE_4_TYPES: Mapping[int, str] = {
    1: "subcutaneous_fat", 2: "torso_fat", 3: "skeletal_muscle", 4: "intermuscular_fat",
}


_FACE: Mapping[int, str] = {1: "face"}


_FACE_MR_AUXILIARY: Mapping[int, str] = {2: "brain", 3: "liver"}


_BRAIN_STRUCTURES: Mapping[int, str] = {
    1: "brainstem",  # + brain_parenchyma
    2: "subarachnoid_space",
    3: "venous_sinuses",  # + dural folds
    4: "septum_pellucidum",
    5: "cerebellum",
    6: "caudate_nucleus",
    7: "lentiform_nucleus",
    8: "insular_cortex",
    9: "internal_capsule",
    10: "ventricle",
    11: "central_sulcus",
    12: "frontal_lobe",
    13: "parietal_lobe",
    14: "occipital_lobe",
    15: "temporal_lobe",
    16: "thalamus",
}


_HEAD_GLANDS_CAVITIES: Mapping[int, str] = {
    1: "eye_left", 2: "eye_right", 3: "eye_lens_left", 4: "eye_lens_right",
    5: "optic_nerve_left", 6: "optic_nerve_right",
    7: "parotid_gland_left", 8: "parotid_gland_right",
    9: "submandibular_gland_right", 10: "submandibular_gland_left",
    11: "nasopharynx", 12: "oropharynx", 13: "hypopharynx",
    14: "nasal_cavity_right", 15: "nasal_cavity_left",
    16: "auditory_canal_right", 17: "auditory_canal_left",
    18: "soft_palate", 19: "hard_palate",
}


_HEADNECK_BONES_VESSELS: Mapping[int, str] = {
    1: "larynx_air", 2: "thyroid_cartilage", 3: "hyoid", 4: "cricoid_cartilage",
    5: "zygomatic_arch_right", 6: "zygomatic_arch_left",
    7: "styloid_process_right", 8: "styloid_process_left",
    9: "internal_carotid_artery_right", 10: "internal_carotid_artery_left",
    11: "internal_jugular_vein_right", 12: "internal_jugular_vein_left",
}


_HEAD_MUSCLES: Mapping[int, str] = {
    1: "masseter_right", 2: "masseter_left",
    3: "temporalis_right", 4: "temporalis_left",
    5: "lateral_pterygoid_right", 6: "lateral_pterygoid_left",
    7: "medial_pterygoid_right", 8: "medial_pterygoid_left",
    9: "tongue",
    10: "digastric_right", 11: "digastric_left",
}


_HEADNECK_MUSCLES: Mapping[int, str] = {
    1: "sternocleidomastoid_right", 2: "sternocleidomastoid_left",
    3: "superior_pharyngeal_constrictor",
    4: "middle_pharyngeal_constrictor",
    5: "inferior_pharyngeal_constrictor",
    6: "trapezius_right", 7: "trapezius_left",
    8: "platysma_right", 9: "platysma_left",
    10: "levator_scapulae_right", 11: "levator_scapulae_left",
    12: "anterior_scalene_right", 13: "anterior_scalene_left",
    14: "middle_scalene_right", 15: "middle_scalene_left",
    16: "posterior_scalene_right", 17: "posterior_scalene_left",
    18: "sterno_thyroid_right", 19: "sterno_thyroid_left",
    20: "thyrohyoid_right", 21: "thyrohyoid_left",
    22: "prevertebral_right", 23: "prevertebral_left",
}


_THIGH_SHOULDER_MUSCLES: Mapping[int, str] = {
    1: "quadriceps_femoris_left", 2: "quadriceps_femoris_right",
    3: "thigh_medial_compartment_left", 4: "thigh_medial_compartment_right",
    5: "thigh_posterior_compartment_left", 6: "thigh_posterior_compartment_right",
    7: "sartorius_left", 8: "sartorius_right",
    9: "deltoid",
    10: "supraspinatus",
    11: "infraspinatus",  # also contains teres_minor
    12: "subscapularis",
    13: "coracobrachial",
    14: "trapezius",
    15: "pectoralis_minor",
    16: "serratus_anterior",
    17: "teres_major",
    18: "triceps_brachii",
}


_LUNG_NODULES: Mapping[int, str] = {1: "lung", 2: "lung_nodules"}


_KIDNEY_CYSTS: Mapping[int, str] = {1: "kidney_cyst_left", 2: "kidney_cyst_right"}


_KIDNEY_CYSTS_AUXILIARY: Mapping[int, str] = {3: "kidney_left", 4: "kidney_right"}


_BREASTS: Mapping[int, str] = {1: "breast"}


_VENTRICLE_PARTS: Mapping[int, str] = {
    1: "ventricle_frontal_horn_left",
    2: "ventricle_occipital_horn_left",
    3: "ventricle_body_left",
    4: "ventricle_temporal_horn_left",
    5: "ventricle_trigone_left",
    6: "ventricle_frontal_horn_right",
    7: "ventricle_occipital_horn_right",
    8: "ventricle_body_right",
    9: "ventricle_temporal_horn_right",
    10: "ventricle_trigone_right",
    11: "third_ventricle",
    12: "fourth_ventricle",
}


_AORTIC_SINUSES: Mapping[int, str] = {
    1: "left_ventricular_outflow_tract",
    2: "right_coronary_cusp",
    3: "left_coronary_cusp",
    4: "non_coronary_cusp",
}


_LIVER_SEGMENTS: Mapping[int, str] = {
    i: f"liver_segment_{i}" for i in range(1, 9)
}


_CRANIOFACIAL_STRUCTURES: Mapping[int, str] = {
    1: "mandible",
    2: "teeth_lower",
    3: "skull",
    4: "head",
    5: "sinus_maxillary",
    6: "sinus_frontal",
    7: "teeth_upper",
}


_TEST: Mapping[int, str] = {1: "carpal"}


CLASS_MAPS: dict[str, Mapping[int, str]] = {
    "total_v1": _TOTAL_V1,
    "total": _TOTAL,
    "total_mr": _TOTAL_MR,
    "total_highres_test": _TOTAL_HIGHRES_TEST,
    "lung_vessels": _LUNG_VESSELS,
    "oculomotor_muscles": _OCULOMOTOR_MUSCLES,
    "covid": _COVID,
    "cerebral_bleed": _CEREBRAL_BLEED,
    "hip_implant": _HIP_IMPLANT,
    "coronary_arteries": _CORONARY_ARTERIES,
    "body": _BODY,
    "body_mr": _BODY,
    "vertebrae_mr": _VERTEBRAE_MR,
    "pleural_pericard_effusion": _PLEURAL_PERICARD_EFFUSION,
    "liver_vessels": _LIVER_VESSELS,
    "vertebrae_body": _VERTEBRAE_BODY,
    "heartchambers_highres": _HEARTCHAMBERS_HIGHRES,
    "appendicular_bones": _APPENDICULAR_BONES,
    "appendicular_bones_mr": _APPENDICULAR_BONES_MR,
    "appendicular_bones_auxiliary": _APPENDICULAR_BONES_AUXILIARY,
    "tissue_types": _TISSUE_TYPES,
    "tissue_types_mr": _TISSUE_TYPES,
    "tissue_4_types": _TISSUE_4_TYPES,
    "face": _FACE,
    "face_mr": _FACE,
    "face_mr_auxiliary": _FACE_MR_AUXILIARY,
    "brain_structures": _BRAIN_STRUCTURES,
    "head_glands_cavities": _HEAD_GLANDS_CAVITIES,
    "headneck_bones_vessels": _HEADNECK_BONES_VESSELS,
    "head_muscles": _HEAD_MUSCLES,
    "headneck_muscles": _HEADNECK_MUSCLES,
    "thigh_shoulder_muscles": _THIGH_SHOULDER_MUSCLES,
    "thigh_shoulder_muscles_mr": _THIGH_SHOULDER_MUSCLES,
    "lung_nodules": _LUNG_NODULES,
    "kidney_cysts": _KIDNEY_CYSTS,
    "kidney_cysts_auxiliary": _KIDNEY_CYSTS_AUXILIARY,
    "breasts": _BREASTS,
    "ventricle_parts": _VENTRICLE_PARTS,
    "aortic_sinuses": _AORTIC_SINUSES,
    "liver_segments": _LIVER_SEGMENTS,
    "liver_segments_mr": _LIVER_SEGMENTS,
    "craniofacial_structures": _CRANIOFACIAL_STRUCTURES,
    "test": _TEST,
}


AVAILABLE_TASKS: tuple[str, ...] = tuple(CLASS_MAPS.keys())


CLASS_MAP_5_PARTS: dict[str, Mapping[int, str]] = {
    "class_map_part_organs": {
        1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
        5: "liver", 6: "stomach", 7: "pancreas",
        8: "adrenal_gland_right", 9: "adrenal_gland_left",
        10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus", 16: "trachea", 17: "thyroid_gland",
        18: "small_bowel", 19: "duodenum", 20: "colon",
        21: "urinary_bladder", 22: "prostate",
        23: "kidney_cyst_left", 24: "kidney_cyst_right",
    },
    "class_map_part_vertebrae": {
        1: "sacrum",
        2: "vertebrae_S1", 3: "vertebrae_L5", 4: "vertebrae_L4",
        5: "vertebrae_L3", 6: "vertebrae_L2", 7: "vertebrae_L1",
        8: "vertebrae_T12", 9: "vertebrae_T11", 10: "vertebrae_T10",
        11: "vertebrae_T9", 12: "vertebrae_T8", 13: "vertebrae_T7",
        14: "vertebrae_T6", 15: "vertebrae_T5", 16: "vertebrae_T4",
        17: "vertebrae_T3", 18: "vertebrae_T2", 19: "vertebrae_T1",
        20: "vertebrae_C7", 21: "vertebrae_C6", 22: "vertebrae_C5",
        23: "vertebrae_C4", 24: "vertebrae_C3", 25: "vertebrae_C2",
        26: "vertebrae_C1",
    },
    "class_map_part_cardiac": {
        1: "heart", 2: "aorta", 3: "pulmonary_vein",
        4: "brachiocephalic_trunk",
        5: "subclavian_artery_right", 6: "subclavian_artery_left",
        7: "common_carotid_artery_right", 8: "common_carotid_artery_left",
        9: "brachiocephalic_vein_left", 10: "brachiocephalic_vein_right",
        11: "atrial_appendage_left",
        12: "superior_vena_cava", 13: "inferior_vena_cava",
        14: "portal_vein_and_splenic_vein",
        15: "iliac_artery_left", 16: "iliac_artery_right",
        17: "iliac_vena_left", 18: "iliac_vena_right",
    },
    "class_map_part_muscles": {
        1: "humerus_left", 2: "humerus_right",
        3: "scapula_left", 4: "scapula_right",
        5: "clavicula_left", 6: "clavicula_right",
        7: "femur_left", 8: "femur_right",
        9: "hip_left", 10: "hip_right",
        11: "spinal_cord",
        12: "gluteus_maximus_left", 13: "gluteus_maximus_right",
        14: "gluteus_medius_left", 15: "gluteus_medius_right",
        16: "gluteus_minimus_left", 17: "gluteus_minimus_right",
        18: "autochthon_left", 19: "autochthon_right",
        20: "iliopsoas_left", 21: "iliopsoas_right",
        22: "brain", 23: "skull",
    },
    "class_map_part_ribs": {
        1: "rib_left_1", 2: "rib_left_2", 3: "rib_left_3", 4: "rib_left_4",
        5: "rib_left_5", 6: "rib_left_6", 7: "rib_left_7", 8: "rib_left_8",
        9: "rib_left_9", 10: "rib_left_10", 11: "rib_left_11", 12: "rib_left_12",
        13: "rib_right_1", 14: "rib_right_2", 15: "rib_right_3", 16: "rib_right_4",
        17: "rib_right_5", 18: "rib_right_6", 19: "rib_right_7", 20: "rib_right_8",
        21: "rib_right_9", 22: "rib_right_10", 23: "rib_right_11", 24: "rib_right_12",
        25: "sternum", 26: "costal_cartilages",
    },
    "test": _TEST,
}


CLASS_MAP_PARTS_MR: dict[str, Mapping[int, str]] = {
    "class_map_part_organs": {
        1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
        5: "liver", 6: "stomach", 7: "pancreas",
        8: "adrenal_gland_right", 9: "adrenal_gland_left",
        10: "lung_left", 11: "lung_right",
        12: "esophagus",
        13: "small_bowel", 14: "duodenum", 15: "colon",
        16: "urinary_bladder", 17: "prostate",
        18: "sacrum", 19: "vertebrae", 20: "intervertebral_discs",
        21: "spinal_cord",
        22: "heart", 23: "aorta",
        24: "inferior_vena_cava", 25: "portal_vein_and_splenic_vein",
        26: "iliac_artery_left", 27: "iliac_artery_right",
        28: "iliac_vena_left", 29: "iliac_vena_right",
    },
    "class_map_part_muscles": {
        1: "humerus_left", 2: "humerus_right",
        3: "scapula_left", 4: "scapula_right",
        5: "clavicula_left", 6: "clavicula_right",
        7: "femur_left", 8: "femur_right",
        9: "hip_left", 10: "hip_right",
        11: "gluteus_maximus_left", 12: "gluteus_maximus_right",
        13: "gluteus_medius_left", 14: "gluteus_medius_right",
        15: "gluteus_minimus_left", 16: "gluteus_minimus_right",
        17: "autochthon_left", 18: "autochthon_right",
        19: "iliopsoas_left", 20: "iliopsoas_right",
        21: "brain",
    },
}


CLASS_MAP_PARTS_HEADNECK_MUSCLES: dict[str, Mapping[int, str]] = {
    "class_map_part_muscles_1": {
        1: "sternocleidomastoid_right", 2: "sternocleidomastoid_left",
        3: "superior_pharyngeal_constrictor",
        4: "middle_pharyngeal_constrictor",
        5: "inferior_pharyngeal_constrictor",
        6: "trapezius_right", 7: "trapezius_left",
        8: "platysma_right", 9: "platysma_left",
        10: "levator_scapulae_right", 11: "levator_scapulae_left",
    },
    "class_map_part_muscles_2": {
        1: "anterior_scalene_right", 2: "anterior_scalene_left",
        3: "middle_scalene_right", 4: "middle_scalene_left",
        5: "posterior_scalene_right", 6: "posterior_scalene_left",
        7: "sterno_thyroid_right", 8: "sterno_thyroid_left",
        9: "thyrohyoid_right", 10: "thyrohyoid_left",
        11: "prevertebral_right", 12: "prevertebral_left",
    },
}


MAP_TASKID_TO_PARTNAME_CT: dict[int, str] = {
    291: "class_map_part_organs",
    292: "class_map_part_vertebrae",
    293: "class_map_part_cardiac",
    294: "class_map_part_muscles",
    295: "class_map_part_ribs",
    517: "test",
}


MAP_TASKID_TO_PARTNAME_MR: dict[int, str] = {
    850: "class_map_part_organs",
    851: "class_map_part_muscles",
}


MAP_TASKID_TO_PARTNAME_HEADNECK_MUSCLES: dict[int, str] = {
    778: "class_map_part_muscles_1",
    779: "class_map_part_muscles_2",
}


COMMERCIAL_MODELS: dict[str, int] = {
    "heartchambers_highres": 301,
    "appendicular_bones": 304,
    "appendicular_bones_mr": 855,
    "tissue_types": 481,
    "tissue_types_mr": 925,
    "tissue_4_types": 485,
    "vertebrae_body": 305,
    "face": 303,
    "face_mr": 856,
    "brain_structures": 409,
    "thigh_shoulder_muscles": 857,
    "thigh_shoulder_muscles_mr": 857,
    "coronary_arteries": 507,
    "aortic_sinuses": 920,
}
"""
Upstream TotalSegmentator commercial-model task IDs. Mirrors the
``commercial_models`` mapping from the legacy ``BioImaging/src/values.py``;
kept here as metadata so downstream code can cross-reference an upstream
task id with its ``CLASS_MAPS`` key.
"""


def get_class_map(task: str) -> Mapping[int, str]:
    """Return the id→name map for *task*."""
    try:
        return CLASS_MAPS[task]
    except KeyError as exc:
        raise ValueError(
            f"Unknown TotalSegmentator task '{task}'. Known: {tuple(CLASS_MAPS.keys())}"
        ) from exc


def get_class_id(class_name: str, task: str) -> int:
    """Return the integer label ID for *class_name* in *task*."""
    m = get_class_map(task)
    for cid, name in m.items():
        if name == class_name:
            return int(cid)
    raise ValueError(f"Class '{class_name}' not found in task '{task}'.")


def get_class_name(class_id: int, task: str) -> str:
    """Return the class name for a numeric *class_id* in *task*."""
    m = get_class_map(task)
    if int(class_id) in m:
        return m[int(class_id)]
    raise ValueError(f"Class id {class_id} not found in task '{task}'.")


__all__ = [
    "CLASS_MAPS",
    "AVAILABLE_TASKS",
    "CLASS_MAP_5_PARTS",
    "CLASS_MAP_PARTS_MR",
    "CLASS_MAP_PARTS_HEADNECK_MUSCLES",
    "MAP_TASKID_TO_PARTNAME_CT",
    "MAP_TASKID_TO_PARTNAME_MR",
    "MAP_TASKID_TO_PARTNAME_HEADNECK_MUSCLES",
    "COMMERCIAL_MODELS",
    "get_class_map",
    "get_class_id",
    "get_class_name",
]
