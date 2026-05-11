<style>
.figure-card {
  border: 1px solid rgb(58, 58, 58);
  border-radius: 10px;
  padding: 14px;
  margin: 18px 0;
  background: #fcfcfc;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.figure-title {
  font-family: sans-serif;
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 8px;
  color: #2c3e50;
}
.figure-subtitle {
  font-family: sans-serif;
  font-size: 13px;
  color: #666;
  margin-bottom: 12px;
}
.figure-caption {
  font-family: sans-serif;
  font-size: 12px;
  color: #444;
  margin-top: 10px;
}
.small-note {
  font-family: sans-serif;
  font-size: 11px;
  color: #666;
}
.label-box {
  fill: #f7f7f7;
  stroke: #d9d9d9;
  stroke-width: 1;
  rx: 8;
  ry: 8;
}
.label-text {
  font-family: sans-serif;
  font-size: 13px;
  fill: #222;
}
.arrow {
  stroke: #555;
  stroke-width: 2.2;
  fill: none;
  marker-end: url(#arrowhead);
}
.soft-red { fill: #e76f51; }
.soft-blue { fill: #5dade2; }
.soft-green { fill: #52be80; }
.soft-gold { fill: #f4d03f; }
.soft-purple { fill: #af7ac5; }
.soft-gray { fill: #95a5a6; }
.soft-pink { fill: #f5b7b1; }
.vessel-red { stroke: #c0392b; fill: none; stroke-width: 7; stroke-linecap: round; }
.vessel-red-thin { stroke: #c0392b; fill: none; stroke-width: 4; stroke-linecap: round; }
.venous-blue { stroke: #2e86c1; fill: none; stroke-width: 6; stroke-linecap: round; }
</style>

> **Date:** 18/03/2026 

> **Data Version:** v0.1.0 [preliminary]

# Project Report: PESA-Brain 4DFlow MRI

<h2 style="text-align: center;">Characterizing the pre-clinical cerebral hemodynamic phenotype associated with Alzheimer's Disease (AD) risk in middle-aged individuals.</h2>

# Contents

1. [Data Description](#overview)
    <!-- - [4DFlow MRI](#4dflow_o)
    - [Time-Of-Flight MRI](#tof_o)
    - [Arterial Spin Labeling Perfusion MRI](#asl_o) -->
2. [Clinical Variables](#clinical_vars)
    - [Clinical Profile](#clinical_profile)
3. [Image Variables](#img_vars)
    - [4DFlow MRI - Macrovasculature Hemodynamic Characterization](#img_vars_4df)
    - [ASL Perfusion MRI - Microvasculature Perfusion Characterization](#img_vars_Asl)
4. [Data Validation](#dval)
    - [QVT Cohort](#dval_qvt)
5. [Hypotesis](#hyp)
    - [Macro-Microvasculature Coupling](#hyp_mmvc)
    - [Hemodynamics vs. Structural Damage](#hyp_hemo_struct_damage)
    - [Hemodynamics vs. Arterial Anatomy](#hyp_hemo_arterial_anatomy)
    - [Hemodynamics vs. Globla Vascular Risk | Systemic Atherosclerosis](#hyp_hemo_vas_risk)
    - [Hemodynamics vs. Cognition](#hyp_hemo_cog)
    - [Hemodynamics vs. Alzheimers Disease Biomarkers](#hyp_hemo_ad)
    - [Predictive value of 4DFLow-derived Markers in Longitudinal Changes](#hyp_pred_long)
    - [Exploratory hypotheses](#hyp_exp)
- [Project Related Papers](#papers)
- [Appendix](#appendix)

---

<a id="overview"></a>

## 1. Data Description 

This report is based on the PESA-Brain imaging cohort, with a specific focus on the subset of participants with imaged 4DFlow MRI. The dataset combines intracranial macrovascular hemodynamics, tissue perfusion, systemic vascular risk markers, and additional clinical markers along structural imaging modalities that together enable a multiscale characterization of cerebrovascular health in midlife.

Because modality availability differs across participants, the absolute counts shown below should be interpreted as modality-specific availability and not as the final sample size for the final multimodal analysis. The effective sample size will depend on the intersection of subjects across the modalities required for each analysis.

📝 **PESA-Brain study design** paper: [Subclinical atherosclerosis and brain health in midlife: Rationale and design of the PESA-Brain study](https://www.sciencedirect.com/science/article/pii/S0002870324002618?via%3Dihub)

<table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
  <thead>
    <tr style="background-color:rgb(90, 90, 90); border-bottom: 2px solid #ddd;">
      <th style="padding: 10px; text-align: left;">Source</th>
      <th style="padding: 10px; text-align: left;">Expected</th>
      <th style="padding: 10px; text-align: left;">Available XNAT</th>
      <th style="padding: 10px; text-align: left;">Missing XNAT</th>
      <th style="padding: 10px; text-align: left;">Processed</th>
      <th style="padding: 10px; text-align: left;">Failed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>PESA-Brain</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">1001</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">1001</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>PESA-Brain Imaging</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">978</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">978</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="6" style="padding: 8px; border-bottom: 1px solid #eee;">Available sources</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>ASL Perfusion</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">961 (+422 IA)</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">961 (+422 IA)</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center; background-color: #c6efce; color: #006100;">961</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>4DFlow MRI</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">554</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">404</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center; background-color: #ffc7ce; color: #9c0006;">150</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center; background-color: #c6efce; color: #006100;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center; background-color: #ffc7ce; color: #9c0006;">48</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>TOF MRI</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">966</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">966</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center; background-color: #c6efce; color: #006100;">404</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>CarotideQFlow MRI</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">844</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="6" style="padding: 8px; border-bottom: 1px solid #eee;">Additional available sources</td>
    </tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>FLAIR</b></td><td>-</td><td>973</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>T1</b></td><td>-</td><td>975</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>T2</b></td><td>-</td><td>976</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>SWI/QSM</b></td><td>-</td><td>969</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>RESTING STATE</b></td><td>-</td><td>921</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>CT-PET</b></td><td>?</td><td>?</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody>
</table>

> ⚠️ **Note:** These *N* are absolute individual numbers, we have to assess the final intersection number of images between modalities.

> ⚠️ **Note:** Check if the 4DFlow MRI *48 Failed QVT+* subjects are retrievable.

In practical terms, the current report is centered on the processed 4DFlow subset and its overlap with ASL and clinical variables. Additional modalities such as TOF, FLAIR, T1, SWI/QSM, and cognition-related variables are relevant for secondary and future analyses, but their final harmonized intersection with the 4DFlow cohort must still be established.

---

<a id="clinical_vars"></a>

## 2. Clinical Variables

The clinical variables were selected to capture the major axes of vascular, Atherosclerotic and Alzheimer-related vulnerability that may influence cerebral hemodynamics in midlife. These include demographic factors, blood pressure, lipid profile, smoking, hematologic variables, global cardiovascular risk scores, systemic atherosclerotic burden, and APOE genotype.

<table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
  <thead>
    <tr style="background-color: rgb(90, 90, 90); text-align: left; border-bottom: 2px solid #ddd;">
      <th style="padding: 10px; text-align: left;">Var</th>
      <th style="padding: 10px; text-align: left; text-align: center;">Units</th>
      <th style="padding: 10px; text-align: left; text-align: center;">N</th>
      <th style="padding: 10px; text-align: left; text-align: center;">mean +- std</th>
      <th style="padding: 10px; text-align: left; text-align: center;">min-max</th>
      <th style="padding: 10px; text-align: left; text-align: center;">5p</th>
      <th style="padding: 10px; text-align: left; text-align: center;">95p</th>
      <th style="padding: 10px; text-align: left; text-align: center;">Desc</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="8" style="padding: 8px; border-bottom: 1px solid #eee;">Basic</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Sex</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">(64.7% Male / 35.3% Female)</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Age at MRI</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Years</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">57.81 +- 4.67</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">49.6-67.7</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">50.72</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">65.16</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">(ImagingDate - BirthDate).days / 365</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Weight</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Kg</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">78.57 +- 15.45</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">44.0-135.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">54.50</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">102.08</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Patient's Weight [kg]</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Height</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Cm</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">170.64 +- 8.25</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">145.0-193.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">156.75</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">182.25</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Patient Height [cm]</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>BMI</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Kg/m2</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">26.86 +- 4.34</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">17.0-43.5</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">19.89</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">34.99</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Weight / (Height / 100)**2 [kg/m2]</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>PSQTO000</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">352</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">1.16 +- 0.98</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-3.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">2.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Smoker status; 0: non; 1: active; 2: former; 3: social</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>LBXHDD</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mg/dL</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">350</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">61.07 +- 14.17</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">28.0-105.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">41.45</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">88.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Cholesterol HDL</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>LBDLDL</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mg/dL</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">347</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">121.49 +- 31.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">8.0-230.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">73.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">173.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Cholesterol LDL</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>LBXTC</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mg/dL</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">350</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">201.17 +- 36.35</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">92.0-316.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">141.45</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">266.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Total Cholesterol</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="8" style="padding: 8px; border-bottom: 1px solid #eee;">Cardiac</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>BPXSYM</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mmHg</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">352</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">126.22 +- 15.86</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">90.0-182.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">102.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">152.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mean(Systolic pressure, 3 measurements)</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>BPXDIM</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mmHg</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">352</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">79.14 +- 9.79</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">53.0-110.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">64.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">96.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mean(Diastolic pressure, 3 measurements)</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>BPXPLS</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">bpm</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">352</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">64.74 +- 10.78</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">39.0-99.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">48.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">84.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Pulse</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>TAS</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mmHg</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">131.70 +- 12.77</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">98.0-180.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">110.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">150.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Systolic pressure at Imaging (Unique Measure)</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>TAD</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mmHg</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">80.41 +- 8.57</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">52.0-105.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">66.75</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">93.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Diastolic pressure at Imaging (Unique Measure)</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>ΔSys/Dias</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mmHg</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">352</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">47.08 +- 9.74</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">26.0-90.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">32.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">62.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">BPXSYM - BPXDIM</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Hematocrit</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">%</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">354</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">44.99 +- 3.89</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">35.0-61.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">39.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">51.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Blood density; % Red cells in blood</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>PEDSCORE2 [V4]</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">350</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">4.00 +- 2.06</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.7-11.6</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">1.27</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">7.53</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">SCORE2 Risk Factor: $1 - S_{10}(t) \exp(\sum \beta_i(X_i - \bar{X}_i))$</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>PEDFRAMI10 [V3]</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">355</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.07 +- 0.05</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-0.3</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.02</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.17</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Framingham Risk Factor (10year)</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>PEDFRAMI30 [V3]</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">331</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.21 +- 0.13</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-0.7</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.06</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.44</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Framingham Risk Factor (30year)</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="8" style="padding: 8px; border-bottom: 1px solid #eee;">Calcium [V4]</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>TACSCTOT</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Agatston Units</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">343</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">66.69 +- 209.64</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-2251.0</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">371.94</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Total Calcium Score</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="8" style="padding: 8px; border-bottom: 1px solid #eee;">Plaque [V3]</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Left Carotid Plaque Vol</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mm3</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">8.73 +- 31.71</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-365.8</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">39.25</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Volume of plaque in Left Carotid</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Right Carotid Plaque Vol</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mm3</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">7.97 +- 30.63</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-461.9</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">41.38</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Volume of plaque in Right Carotid</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Total Carotid Plaque Vol</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mm3</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">16.70 +- 49.96</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-530.2</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">76.77</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Combined Left and Right Carotid Volume</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Total Femoral Plaque Vol</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mm3</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">60.11 +- 141.36</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-1122.2</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">278.81</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Total volume in Femoral arteries</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>Total Plaque Vol</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">mm3</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">356</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">76.81 +- 163.15</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.0-1137.8</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">0.00</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">326.63</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Cumulative Plaque Volume across all sites</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="8" style="padding: 8px; border-bottom: 1px solid #eee;">White Matter (FLAIR)</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>-</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="background-color: rgba(0, 0, 0, 0); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
    </tr>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="8" style="padding: 8px; border-bottom: 1px solid #eee;">Genetic</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>APOE</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Genotype</td>
      <td style="background-color: rgb(162, 98, 1); padding: 8px; border-bottom: 1px solid #eee; text-align: center;">354</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">Apolipoprotein E Genotype status</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="8" style="padding: 8px; border-bottom: 1px solid #eee;">Cognitives</td>
    </tr>
    <tr style="background-color: rgb(135, 0, 0);">
      <td style="padding: 8px; border-bottom: 1px solid #eee;"><b>TBA</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;">-</td>
    </tr>
  </tbody>
</table>

> ⚠️ **Note:** This profile is computed over the 356 available subjects with processed 4DFlows.

> ⚠️ **Cognitive variables yet to be described and reported.**

<a id="clinical_profile"></a>

### 2.2. Clinical Profile

The following distributions provide a descriptive overview of the <b>already processed</b> 4DFlow cohort. Their purpose is not only to summarize cohort composition, but also to contextualize the expected range and heterogeneity of the vascular factors that may shape intracranial hemodynamics.

<!-- ```{=html} -->
<div style="display: flex; flex-wrap: wrap; justify-content: flex-start; font-family: sans-serif; gap: 15px;">

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_sex.png" style="width: 75%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Sex Distribution</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_age.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Age at MRI</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_weight.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Weight</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_height.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Height</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_bmi.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">BMI</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_psqto000.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">PSQTO000 (Smoking)</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_lbxhdd.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Cholesterol HDL</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_lbdldl.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Cholesterol LDL</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_lbxtc.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Total Cholesterol</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_bpxsym.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">BPXSYM (Systolic)</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_bpxdim.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">BPXDIM (Diastolic)</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_bpxpls.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Pulse</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_tas.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">TAS (Imaging Systolic)</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_tad.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">TAD (Imaging Diastolic)</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_sys_dyas_delta.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">ΔSys/Dias</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_hematocrit.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Hematocrit</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_pedscore2.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">PEDSCORE2 [V4]</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_pedframi10.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">PEDFRAMI10 [V3]</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_pedframi30.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">PEDFRAMI30 [V3]</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_tacsctot.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">TACSCTOT (Calcium)</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_lca_plc.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Left Carotid Plaque Vol</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_rca_plc.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Right Carotid Plaque Vol</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_total_carotide_plc.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Total Carotid Plaque Vol</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_total_femoral_plc.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Total Femoral Plaque Vol</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_total_plc.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">Total Plaque Vol</p>
  </div>

  <div style="width: 28%; border: 1px solid #eee; padding: 10px; margin-bottom: 10px; text-align: center; border-radius: 5px;">
    <img src="./plots/report/f_apoe.png" style="width: 100%; height: auto;">
    <p style="font-size: 11px; font-weight: bold; margin-top: 5px;">APOE Genotype</p>
  </div>

</div>
<!-- ``` -->

> ⚠️ **Note:** This profile is computed over the 356 available subjects with processed 4DFlows.

---

<a id="img_vars"></a>

### 3. Image Variables

The imaging variables are organized according to the biological scale they describe. 4DFlow MRI captures macrovascular hemodynamics in the large intracranial arteries and venous sinuses; ASL captures tissue-level perfusion; and TOF or structural MRI extensions provide complementary information on vascular anatomy and downstream tissue injury. 

<a id="img_vars_4df"></a>

#### 3.1. 4DFlow 

The 4DFlow-derived variables can be understood as belonging to four main families: bulk flow delivery, pulsatile waveform behavior, stiffness/transmission properties, and exploratory descriptors. Together, these metrics aim to characterize not only how much blood enters the brain, but also how pulsatile energy propagates through the intracranial vascular system.

<table style="border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 11px; border: 1px solid #ddd;">
<thead>
<tr style="background-color: #5A5A5A; color: white;">
<th style="padding: 10px; text-align: left; border: 1px solid #ddd; width: 25%;">Category</th>
<th style="padding: 10px; text-align: left; border: 1px solid #ddd; width: 25%;">Parameter</th>
<th style="padding: 10px; text-align: left; border: 1px solid #ddd; width: 30%;">Vessels / ROI</th>
<th style="padding: 10px; text-align: left; border: 1px solid #ddd; width: 20%;">Notes / Formulas</th>
</tr>
</thead>
<tbody>
<tr style="background-color: #f2f2f2; font-weight: bold;">
<td colspan="4" style="padding: 8px; border: 1px solid #ddd;">1. Basic 4D Flow Metrics</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><b>Localized time-Avgerage Blood Flow [mL/s]</b></td>
<td style="padding: 8px; border: 1px solid #ddd;">Total & Vessel-Specific</td>
<td style="padding: 8px; border: 1px solid #ddd;"><b>Arterial:</b> ICAs, MCAs, Basilar, PCAs<br><b>Venous:</b> SSS, STR Sinus, LTRV Sinus, RTRV Sinus</td>
<td style="padding: 8px; border: 1px solid #ddd;">LOC specific cross-sectional flow mean across the 15 cardiac frames<br>Total = ∑(ICAs+Basilar)</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><b>Localized timseries Blood Flow [15f] [mL/s]</b></td>
<td style="padding: 8px; border: 1px solid #ddd;">Total & Vessel-Specific</td>
<td style="padding: 8px; border: 1px solid #ddd;"><b>Arterial:</b> ICAs, MCAs, Basilar, PCAs<br><b>Venous:</b> SSS, STR Sinus, LTRV Sinus, RTRV Sinus</td>
<td style="padding: 8px; border: 1px solid #ddd;">LOC specific cross-sectional flow at all 15 cardiac frames<br>Total = ∑(ICAs+Basilar)</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><b>Functional Luminal Area</b></td>
<td style="padding: 8px; border: 1px solid #ddd;">Vessel-Specific</td>
<td style="padding: 8px; border: 1px solid #ddd;"><b>Arterial:</b> ICAs, MCAs, Basilar, PCAs<br><b>Venous:</b> SSS, STR Sinus, LTRV Sinus, RTRV Sinus</td>
<td style="padding: 8px; border: 1px solid #ddd;">LOC specific cross-sectional area during cardiac cycle</td>
</tr>

<tr style="background-color: #f2f2f2; font-weight: bold;">
  <td colspan="4" style="padding: 8px; border: 1px solid #ddd;">2. Blood Velocity</td>
</tr>
<tr>
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Peak Systolic Velocity [PSV]</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Mean & Vessel-Specific</td>
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Arterial:</b> ICAs, MCAs, Basilar, PCAs<br><b>Venous:</b> SSS, STR Sinus, LTRV Sinus, RTRV Sinus</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Maximum velocity at peak systole</td>
</tr>
<tr style="background-color:rgba(171, 0, 0, 0.07)">
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Pulse Wave Velocity [PWV] (Stiffness metric)</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;"> Mean & Vessel-Specific</td>
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Arterial:</b> ICAs, MCAs, Basilar, PCAs<br><b>Venous:</b> SSS, STR Sinus, LTRV Sinus, RTRV Sinus</td>
  <td style="padding: 8px; border: 1px solid #ddd;">∀Vessel $LOC_{ini}$ & $LOC_{fin}$<br>$PWV = \frac{Δx}{Δt}$<br>Where Δx(Dist.): Centerline distance between init and fin; Δt(tDelay): Time offset between the arrival of the systolic peak point init vs. fin</td>
</tr>

<tr style="background-color: #f2f2f2; font-weight: bold;">
  <td colspan="4" style="padding: 8px; border: 1px solid #ddd;">3. Pulsatility & Resistance Indices</td>
</tr>
<tr>
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Pulsatility (PI) & Resistance (RI)</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Vessel-Specific</td>
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Arterial:</b> ICAs, MCAs, Basilar, PCAs<br><b>Venous:</b> SSS, STR Sinus, LTRV Sinus, RTRV Sinus</td>
  <td style="padding: 8px; border: 1px solid #ddd;">$PI = \frac{V_{max} - V_{min}}{V_{mean}}$<br>With $V_{mean}$ across all 15 cardiac frames & <br>$RI = \frac{V_{max} - V_{min}}{V_{max}}$</td>
</tr>
<tr>
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Intracranial Pulsatility Bourden [IPB]</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Derived Ratios</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Arterial</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Ratio: $Mean(ICAs, Basi)$</td>
</tr>
<tr>
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Arterial to Venous Pulsatility Bourden [A2VPB]</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Derived Ratios</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Arterial & Venous coupling</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Ratio: $\frac{\text{Mean Arterial PI}}{\text{Mean Venous PI}}$</td>
</tr>
<tr>
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Anterior vs Posterior Circulation Pulsatility Index [APCPI]</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Derived Ratios</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Arterial Anterior & Posterior Circulation</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Ratio: $\frac{\text{Mean Arterial PI}}{\text{Mean Venous PI}}$</td>
</tr>
<tr style="background-color:rgba(171, 0, 0, 0.07)">
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Pulsatility Index Transmission Ciefficient [PITC | pt]</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Pulsatility Transmission Indices</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Arterial tree</td>
  <td style="padding: 8px; border: 1px solid #ddd;">The paper define a Pulsatility Transmission Function where PI is tracked along the vessel centerline relative to the distance ($d$) from a root vesel $p(d) = p_t​ ⋅ d + β$ <br> Where $p_t$: Slope of the PI distribution along vessel points; $β$: PI at root</td>
</tr>
<tr style="background-color:rgba(171, 0, 0, 0.07)">
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Dampling Index [pd]</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Pulsatility Transmission Indices</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Arterial tree</td>
  <td style="padding: 8px; border: 1px solid #ddd;">$p_d = \frac{PI_{proximal}-PI_{distal}}{PI_{proximal}}$ <br> The paper notes that pd​ is a better representation of vascular compliance and the Windkessel effect than measuring PI alone, because PI is often biased by the specific vessel anatomy, whereas pd​ measures the change between two states.</td>
</tr>
<tr style="background-color: #f2f2f2; font-weight: bold;">
  <td colspan="4" style="padding: 8px; border: 1px solid #ddd;">4. Exploratory & Advanced Parameters</td>
</tr>
<tr style="background-color:rgba(171, 0, 0, 0.07)">
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Wall Shear Stress [WSS]</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Friction Metric</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Large caliber segments</td>
  <td style="padding: 8px; border: 1px solid #ddd;">WSS is the tangential force exerted by the flowing blood on the endothelial cells (the inner lining of the artery) <br> $τ_w ​= μ ⋅ (\frac{∂n}{∂u​})_{wall}$ <br>​ Where μ: Blood Viscosity; $\frac{∂n}{∂u​}$: Velocity Gradient (Change in speed as you move from the vall toward the ceter of the vessel) </td>
</tr>
<tr style="background-color:rgba(171, 0, 0, 0.07)">
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Oscilatory Shear Index [OSI]</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Friction Metric</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Large caliber segments</td>
  <td style="padding: 8px; border: 1px solid #ddd;">OSI measures the net directional change of blood-induced friction over a cardiac cycle, where higher values (up to 0.5) identify regions of "sloshing" flow prone to plaque formation. <br> $OSI = 0.5 \cdot \left( 1 - \frac{\left| \int_{0}^{T} \vec{\tau}_w dt \right|}{\int_{0}^{T} |\vec{\tau}_w| dt} \right)$ <br>​ Where Numerator(Net.): This is the magnitude of the vector sum. If the stress points forward for half the cycle and backward for the other half, they cancel each other out, and this value becomes zero; Denominator(Total): This is the sum of the magnitudes. It doesn't care about direction; it just measures the total amount of "rubbing" the wall felt, regardless of which way it was pointing. </td>
</tr>
<tr style="background-color:rgba(171, 0, 0, 0.07)">
  <td style="padding: 8px; border: 1px solid #ddd;"><b>Waveform Metrics</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Total & Vessel-Specific</td>
  <td style="padding: 8px; border: 1px solid #ddd;">-</td>
  <td style="padding: 8px; border: 1px solid #ddd;">(e.g. Time to systolic peak, Low Frequency Oscilations, ...)</td>
</tr>
<tr style="background-color:rgba(171, 0, 0, 0.07)">
  <td style="padding: 8px; border: 1px solid #ddd;"><b>ML Phenotypes</b></td>
  <td style="padding: 8px; border: 1px solid #ddd;">Hemodynamic Clusters</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Full 4D Flow dataset</td>
  <td style="padding: 8px; border: 1px solid #ddd;">Unsupervised feature grouping</td>
</tr>

</tbody>
</table>

<a id="img_vars_asl"></a>

#### 3.2. ASL Perfusion 

ASL-derived perfusion measures complement 4DFlow by shifting the focus from flow in major vessels to tissue-level blood delivery. These measures are particularly useful for testing whether macrovascular abnormalities are accompanied by regional hypoperfusion patterns, including general, territory-specific or atlas-based effects.

<table style="border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 11px; border: 1px solid #ddd;">
<thead>
<tr style="background-color: #5A5A5A; color: white;">
<th style="padding: 10px; text-align: left; border: 1px solid #ddd; width: 25%;">Category</th>
<th style="padding: 10px; text-align: left; border: 1px solid #ddd; width: 25%;">Parameter</th>
<th style="padding: 10px; text-align: left; border: 1px solid #ddd; width: 30%;">Vessels / ROI</th>
<th style="padding: 10px; text-align: left; border: 1px solid #ddd; width: 20%;">Notes / Formulas</th>
</tr>
</thead>
<tbody>
<tr style="background-color: #f2f2f2; font-weight: bold;">
<td colspan="4" style="padding: 8px; border: 1px solid #ddd;">1. Basic ASL Perfusion Metrics - Cerebral Blood Flow [mL/100g/min]</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><b>CBF</b></td>
<td style="padding: 8px; border: 1px solid #ddd;">Total & Hemisphere-Specific</td>
<td style="padding: 8px; border: 1px solid #ddd;"><b>Total</b> & <b>Hemi.:</b> R/L</td>
<td style="padding: 8px; border: 1px solid #ddd;">Mean Celebral Blood Flow Pixel-wise</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><b>CBF Desikan-labeled</b></td>
<td style="padding: 8px; border: 1px solid #ddd;">Cerebral Region-Specific</td>
<td style="padding: 8px; border: 1px solid #ddd;">Desikan-Specific Atlas regions</td>
<td style="padding: 8px; border: 1px solid #ddd;">Mean Celebral Blood Flow Pixel-wise</td>
</tr>
<tr>
<td style="padding: 8px; border: 1px solid #ddd;"><b>CBF Vascular-labeled</b></td>
<td style="padding: 8px; border: 1px solid #ddd;">Cerebral Region-Specific</td>
<td style="padding: 8px; border: 1px solid #ddd;">Vasular-Specific Atlas regions</td>
<td style="padding: 8px; border: 1px solid #ddd;">Mean Celebral Blood Flow Pixel-wise</td>
</tr>
</tbody>
</table>

---

<a id="dval"></a>

### 4. Data Validation

This validation step assesses whether the main flow and pulsatility patterns derived from the PESA-Brain cohort are broadly consistent with previously reported normative 4DFlow findings. The goal is not to establish exact equivalence between cohorts, but to determine whether the observed distributions, age-related trends, and vessel-wise behaviors are directionally coherent and biologically plausible.

📝 **QVT Normative Hemodynamics in midle-agled and older adults** paper: [Normative Cerebral Hemodynamics in Middle-aged and Older Adults Using 4D Flow MRI: Initial Analysis of Vascular Aging](https://pmc.ncbi.nlm.nih.gov/articles/PMC10140641/)

<!-- ```{=html} -->
<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
  <div style="flex: 1; max-width: 70%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="./plots/report_validation/diff_erosion.png" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      Figure A: Flow means QVT Cohort vs. PESA-Brain 356 Cohort | Diff. mean Flows when eroding mask 
    </p>
  </div>
  <div style="flex: 1; max-width: 25%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="./plots/report_validation/table_1.png" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      Figure B: QVT Cohort Clinical Review
    </p>
  </div>
</div>

<div style="font-family: sans-serif; max-width: 100%; margin: 18px 0;">
  <div style="padding: 14px 18px; border-left: 4px solid #4c78a8; background:rgb(41, 41, 41); margin-bottom: 14px;">
    <div style="font-weight: 700; margin-bottom: 6px;">Flow Model</div>
    <div style="font-size: 22px;">
      <i>flow</i> ~ 1 + age_c + sex + (1 + age_c | vessel) + (1 | participant)
    </div>
  </div>

  <div style="padding: 14px 18px; border-left: 4px solid rgb(91, 168, 76); background: rgb(41, 41, 41); margin-bottom: 14px;">
    <div style="font-weight: 700; margin-bottom: 6px;">Pulsatility Model</div>
    <div style="font-size: 22px;">
      log(<i>PI</i>) ~ 1 + age_c + sex + (1 + age_c | vessel) + (1 | participant)
    </div>
  </div>
</div>

<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
<div style="flex: 1; max-width: 100%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
  <img src="./plots/report_validation/figure_m_flow.png" style="width: 100%; height: auto; border-radius: 4px;">
  <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
    Figure C.I: QVT Cohort Linear Mixed Model [LOC tAvg Flow]
  </p>
</div>
</div>

<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
<div style="flex: 1; max-width: 100%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
  <img src="./plots/report_validation/our_m_flow.png" style="width: 100%; height: auto; border-radius: 4px;">
  <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
    Figure C.II: PESA-Brain Cohort Linear Mixed Model [LOC tAvg Flow]
  </p>
</div>
</div>

<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
<div style="flex: 1; max-width: 100%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
  <img src="./plots/report_validation/fig_diff_pi.png" style="width: 100%; height: auto; border-radius: 4px;">
  <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
    Figure C.II: PESA-Brain Cohort Linear Mixed Model [PI]
  </p>
</div>
</div>
<!-- ``` -->

<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">

  <h2 style="color: #27ae60; border-bottom: 1px solid rgb(50, 50, 50); padding-bottom: 10px;">Key Findings: QVT Cohort validation</h2>

  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Flow & PI Distributions:</strong> 
     Both data distributions perfectly aligns. <br>
     ⚠️ There is a difference in the Flow scale.
  </div>
  
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Flow Scale:</strong> 
     The scale discrepancy can be explained and measured by the resolution missmatch, affecting the PESA-Brain cohort's lower 4DFlow MRI resolutions.
  </div>

  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Age effect on Flow and PI:</strong> 
     Both cohorts coincide in a decrease of flow and increase of puslatility with aging, while PESA-Brain proposes an slighly higher decrease per of flow year (And aligning on the pulsatility increase), this differece can be explained by the clinical profile difference between cohorts and age range difference.
  </div>

</div>

<!-- ```{=html} -->
<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">

  <h2 style="color: #27ae60; border-bottom: 1px solid rgb(50, 50, 50); padding-bottom: 10px;">Additional Key Findings: </h2>

  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Hmatocrit:</strong> 
    If we correct the model by hematoctit the age effect and significance becomes null while the hematocrite effects has a strong significance on the model's effect. <br>
    Model: <i>flow</i> ~ 1 + age_c + sex + hematocrit + (1 + age_c | vessel) + (1 | participant)
    <div style="flex: 1; max-width: 100%; text-align: center; border: 1px solid #eee; padding: 10px; border-radius: 8px;">
        <img src="./plots/report_validation/table_flo_hematocrit.png" style="width: 75%; height: auto; border-radius: 4px;">
        <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
          Table A: Flow params from the Linear Mixed-Effects Model correcting by Hematocrite. 
        </p>
      </div>
  </div>

  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Flow decrease & PI increase Vessel-wise:</strong> 
    <div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
    <div style="flex: 1; max-width: 70%; text-align: center; border: 1px solid #eee; padding: 10px; border-radius: 8px;">
        <table style="border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 13px; border: 1px solid #ddd; margin: 20px 0;">
          <thead>
          <tr style="background-color:rgb(68, 68, 68); color: white; text-align: left;">
          <th style="padding: 12px; border: 1px solid #ddd;">Vessel</th>
          <th style="padding: 12px; border: 1px solid #ddd;">Flow Intercept</th>
          <th style="padding: 12px; border: 1px solid #ddd;">Flow Age Slope</th>
          <th style="padding: 12px; border: 1px solid #ddd;">PI Intercept</th>
          <th style="padding: 12px; border: 1px solid #ddd;">PI Age Slope</th>
          </tr>
          </thead>
          <tbody>
          <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Basilar</td>
          <td style="padding: 8px; border: 1px solid #ddd;">179.312</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-1.571</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.262</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.008</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Left_ACA</td>
          <td style="padding: 8px; border: 1px solid #ddd;">106.983</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-1.478</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.206</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.013</td>
          </tr>
          <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Left_ICA</td>
          <td style="padding: 8px; border: 1px solid #ddd;">303.075</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-2.593</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.255</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.012</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Left_MCA</td>
          <td style="padding: 8px; border: 1px solid #ddd;">171.753</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-1.959</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.230</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.014</td>
          </tr>
          <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Left_PCA</td>
          <td style="padding: 8px; border: 1px solid #ddd;">76.275</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-0.251</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.130</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.011</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Left_Transverse</td>
          <td style="padding: 8px; border: 1px solid #ddd;">188.506</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-0.375</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.790</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.021</td>
          </tr>
          <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Right_ACA</td>
          <td style="padding: 8px; border: 1px solid #ddd;">102.749</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-1.230</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.162</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.016</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Right_ICA</td>
          <td style="padding: 8px; border: 1px solid #ddd;">291.145</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-2.956</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.227</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.014</td>
          </tr>
          <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Right_MCA</td>
          <td style="padding: 8px; border: 1px solid #ddd;">175.837</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-2.003</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.247</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.014</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Right_PCA</td>
          <td style="padding: 8px; border: 1px solid #ddd;">81.167</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-0.974</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.148</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.016</td>
          </tr>
          <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Right_Transverse</td>
          <td style="padding: 8px; border: 1px solid #ddd;">287.751</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-2.956</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.823</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.011</td>
          </tr>
          <tr style="background-color: #f9f9f9;">
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Sagittal_Sinus</td>
          <td style="padding: 8px; border: 1px solid #ddd;">300.738</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #c0392b;">-2.559</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.900</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.014</td>
          </tr>
          <tr>
          <td style="padding: 8px; border: 1px solid #ddd; font-weight: bold;">Straight_Sinus</td>
          <td style="padding: 8px; border: 1px solid #ddd;">90.988</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #2980b9;">0.577</td>
          <td style="padding: 8px; border: 1px solid #ddd;">-0.807</td>
          <td style="padding: 8px; border: 1px solid #ddd; color: #27ae60;">0.014</td>
          </tr>
          </tbody>
          </table>
        <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
          Table B: Flow and PI Interpets at age = mean and Slopes effects per vessel. 
        </p>
      </div>
  </div>

</div>

<!-- ``` -->
---


<a id="hyp"></a>

### 5. Hypotesis

This section summarizes the main scientific questions that can be addressed with the current PESA-Brain 4DFlow dataset and its multimodal extensions. The main goal is to find and characterize the preclinical cerebral hemodynamic phanotype associated with Alzheimer's Diseade (AD) risk in middle-aged individuals. 4DFlow Brain MRI is capable od providing parameters that inform about the brain circulation macrovascular function and its global pulsatile behaviour and vessel stiffness. 

<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">
  <h2 style="color: #27ae60; border-bottom: 1px solid #27ae60; padding-bottom: 10px;">Previously reported: 4DFlow Brain Studies</h2>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Flow & Stifness:</strong> 
     Patients with <b>AD and Mild Cognitive Impairment (MCI)</b> show <b><a style="color:red;">reduced</a> intracranial flow</b>, <b><a style="color:green;">increased</a> arterial and venous stiffness</b>, <b><a style="color:green;">faster</a> arterial-to-venous pulse transmission and <b><a style="color:red;">reduced</a> damping. <br>
     Separate work showed <a style="color:green;">higher</a> intracranial PWV in AD dementia clinical syndrome when compared to age-matched controls.<br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7476069/">Intracranial vascular flow oscillations in Alzheimer’s disease from 4D flow MRI</a>] <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8719432/">Cerebrovascular stiffness and flow dynamics in the presence of amyloid and tau biomarkers</a>] 
  </div>
  
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Symptomatic stages:</strong> 
     Also in <b>symptomatic stages</b> of the desiase, 4DFlow parameters assiciate with <b>Cognitive Imairment</b>, <b>Amyliod Burden</b>, <b>Tau Burden</b> and MRI markers of small vessel dissease, such as <b>White Matter Hiperintensities</b>. <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5617115/">Intracranial arterial 4D-flow in individuals with Mild Cognitive Impairment is associated with cognitive performance and amyloid positivity</a>] <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8719432/">Cerebrovascular stiffness and flow dynamics in the presence of amyloid and tau biomarkers</a>] <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9266739/#sec5-ijms-23-07106">Vascular Dysfunction Is Central to Alzheimer’s Disease Pathogenesis in APOE e4 Carriers</a>] <br>
  </div>
</div>

However, it is still largely unknown whether 4DFlow Brain MRIis capable of <b>detecting early hbodynamic changes</b> in the preclinical phased of the AD. In this setting, we aim to study 4DFlow Brian MRI in a <b>cohort of middle-aged cognitively healthy individuals</b>, in order to <b>characterize the preclinical cerebral hbodynamic phanotype associated with AD risk</b>.

<a id="hyp_mmvc"></a>

### I. Macro-Microvasculature Coupling

---

Study the relatonship between brain <b>4DFLow MRI</b> hemodynamic parameters (Macrovascular large intercranial arteries) and brain <b>ASL perfusion</b> MRI parameters (Microvascular perfusion). Globally and regionally, while studying which large-vessel hemodynamics affects which cerebral regions.

Working hypothesis
:  Altered large-vessel intracranial hemodynamics, specially <a style="color:green;">lower</a> blood flow and <a style="color:red;">higer</a> pulsatility/stiffness, will be associated/correnated with <a style="color:red;">reduced</a> cerebral perfusion.

Possible expected patterns:
- [ ] Reduced Flow will relate with lower global and regional CBF.
- [ ] Increased PI, PWV, IPB or weaker damping/transmission control will relate with lower global and regional CBF.
---
- [ ] This assicuation may be stronger in different territories, where microvascular vulnerability may be greater (Watershed, Posterior circulation, ...?)
- [ ] Perfusion-Metabolism missmatch; Comparing 4DFlow/ASL with glucose PET metabolism patterns to identify regions where flow may not match the necessary metabolism demand?

<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">
  <h2 style="color: #27ae60; border-bottom: 1px solid #27ae60; padding-bottom: 10px;">Previously reported:</h2>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Micro-Macrovascular coupling:</strong> 
    In congnitively healthy asymptomatic adults at risk for AD, <a style="color:green;">higer</a> large-artery flow in the ICAs and MCAs (Anterior circulation) was positively associated with <a style="color:green;">higer</a> ASL perfusion across broad cortical regions. <br>
    Additionally, <a style="color:red;">reduced</a> perfusion CBF or Hypoperfusion, may be an early marker of neurogeneration that initiates a cascade of events preceding cognitive decline end AD. <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5318539/">Macrovascular and microvascular cerebral blood flow in adults at risk for Alzheimer's disease</a>]
  </div>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Posterior circulation vulnerability:</strong> 
    APOE4 carriers showed brain changes consistent with vascular dysfunction, such as reduced white matter integrity in posterior brain regions and greater effect on amyloid than tau load. <br>
    Other sources suggests that oxigenation, blood flow and glucose uptake decline in the precuneus and posterior cingulate at a very early stage of AD. <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9266739/">Vascular Dysfunction Is Central to Alzheimer’s Disease Pathogenesis in APOE e4 Carriers</a>] <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4982069/">Pathophysiology of Hypoperfusion of the Precuneus in Early Alzheimer's Disease</a>]
  </div>
</div>


Results
---

<u>R1. Whole brain perfusion coupling:</u>

Linear Mixel-Effects Model coupling Flow ~ Whole brain perfusion

<div style="font-family: sans-serif; max-width: 100%; margin: 18px 0;">
  <div style="padding: 14px 18px; border-left: 4px solid #4c78a8; background:rgb(41, 41, 41); margin-bottom: 14px;">
    <div style="font-weight: 700; margin-bottom: 6px;">Whole Brain Model</div>
    <div style="font-size: 22px;">
      <i>flow</i> ~ 1 + ctw_wb + age_c + sex + (1 | vessel) + (1 | participant)
    </div>
  </div>
</div>

<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
  <div style="flex: 1; max-width: 35%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="./plots/report_asl/ctx_wb_model.png" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      Model params for <i>flow</i> ~ 1 + ctw_wb + age_c + sex + (1 | vessel) + (1 | participant)
    </p>
  </div>
  <div style="flex: 1; max-width: 75%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="./plots/report_asl/ctx_wb_model_fits.png" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      Model fits
    </p>
  </div>
</div>

<u>R2. Hemisphere aware perfusion coupling:</u>

Linear Mixel-Effects Model coupling Flow ~ Hemisphere perfusion

<div style="font-family: sans-serif; max-width: 100%; margin: 18px 0;">
  <div style="padding: 14px 18px; border-left: 4px solid rgb(90, 168, 76); background:rgb(41, 41, 41); margin-bottom: 14px;">
    <div style="font-weight: 700; margin-bottom: 6px;">Hemisphere Aware Model</div>
    <div style="font-size: 22px;">
      <i>flow</i> ~ 1 + ctw_h + age_c + sex + (1 | vessel) + (1 | participant)
    </div>
  </div>
</div>

<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
  <div style="flex: 1; max-width: 35%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="./plots/report_asl/ctx_h_model.png" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      Model params for <i>flow</i> ~ 1 + ctw_h + age_c + sex + (1 | vessel) + (1 | participant)
    </p>
  </div>
  <div style="flex: 1; max-width: 75%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="./plots/report_asl/ctx_h_model_fits.png" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      Model fits
    </p>
  </div>
</div>

<u>R3. Cerebral-territories perfusion coupling: (Desikan) </u>


<div style="font-family: sans-serif; max-width: 100%; margin: 18px 0;">
  <div style="padding: 14px 18px; border-left: 4px solid rgb(168, 76, 76); background:rgb(41, 41, 41); margin-bottom: 14px;">
    <div style="font-weight: 700; margin-bottom: 6px;">Cerebral-territories Model</div>
    <div style="font-size: 22px;">
      <i>flow</i> ~ 
    </div>
  </div>
</div>

<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
  <div style="flex: 1; max-width: 35%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      ...
    </p>
  </div>
  <div style="flex: 1; max-width: 75%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      ...
    </p>
  </div>
</div>

<u>R3. Vascular-territories perfusion coupling: </u>


<div style="font-family: sans-serif; max-width: 100%; margin: 18px 0;">
  <div style="padding: 14px 18px; border-left: 4px solid rgb(122, 76, 168); background:rgb(41, 41, 41); margin-bottom: 14px;">
    <div style="font-weight: 700; margin-bottom: 6px;">Vascular-territories Model</div>
    <div style="font-size: 22px;">
      <i>flow</i> ~ 
    </div>
  </div>
</div>

<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
  <div style="flex: 1; max-width: 35%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      ...
    </p>
  </div>
  <div style="flex: 1; max-width: 75%; text-align: center; border: 1px solid rgb(50, 50, 50); padding: 10px; border-radius: 8px;">
    <img src="" style="width: 100%; height: auto; border-radius: 4px;">
    <p style="font-size: 12px; font-weight: bold; margin-top: 10px; color: #333;">
      ...
    </p>
  </div>
</div>


<a id="hyp_hemo_struct_damage"></a>

### II. Hemodynamics vs. Structural Damage

---

Study the relationship between brain <b>4DFLow MRI</b> hemodynamic parameters and MRI biomarkers of <b>Cerebrovascular</b> and <b>Neurodegeneration-driven</b> brain structural damage.

Working hypothesis
:  Adverse large-vessel intracranial hemodynamics, will be associated with MRI markers of brain damage.

Possible structural damage markers:
- [ ] <b>T1/T2:</b> Brain volumes (Hippocampal vol, cerebral atropy...)
- [ ] <b>FLAIR:</b> White matter intensities, Microangiopathic injury, ...
- [ ] <b>DTI tractography:</b> White matter ultrastructure (Microstructural integrity, ...)

<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">
  <h2 style="color: #27ae60; border-bottom: 1px solid #27ae60; padding-bottom: 10px;">Previously reported:</h2>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Structural damage:</strong> 
    The broader 4DFlow/amall-vessel literature links higher intracranial pulsatility with White Matter Hiperintensities (WMH), brain atrophy, enlarged perivascular spaces and hippocampal microvascular disfunction. <br>
     [Source: <a href="https://www.nature.com/articles/s41598-024-63312-4">Measuring global cerebrovascular pulsatility transmission using 4D flow MRI </a>] <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8217890/">Cerebral arterial pulsatility is linked to hippocampal microvascular function and episodic memory in healthy older adults </a>]
  </div>
</div>

<a id="hyp_hemo_arterial_anatomy"></a>

### III. Hemodynamics vs. Arterial Anatomy

---

Study the relationship between brain <b>4DFLow MRI</b> hemodynamic parameters and <b>TOF MRI</b>-derived parameters related with the intracranial arterial structure/anatomy.

Working hypothesis
:  Intracranial arterial anatomy modifies cerebral hemodynamics, and unfavorable vascular geometry will be associated with abnormal flow and pulsatility trnasmisions.

Possible structural/anatomical markers:
- [ ] <b>Intracranial atherosclerosis</b>
- [ ] <b>Intracranial stenosis</b>
- [ ] <b>Anatomic variations (Circle of Willis)</b>
- [ ] <b>Arterial Turtuosity index</b>
- [ ] <b>Intracranial Arterial Calcification</b> -> CT Derived
- [ ] <b>Anatomical asymetries</b>


<a id="hyp_hemo_vas_risk"></a>

### IV. Hemodynamics vs. Globla Vascular Risk | Systemic Atherosclerosis

---

Study the relationship between <b>Global Vascular Risk</b> and <b>Systemic Atherosclerosis</b>.

Working hypothesis
:  Higher cardiovascular risk and higher systemic atherosclerotic burden will be associated with more impaired hemodynamic phenotype.

Possible expected patterns:
- [ ] <a style="color:green;">Higer</a> SCORE2, FRAMINGHAM Scores, Blood pressure, Calcium Score, Plaque burden, ... --relate--> <a style="color:red;">Lower</a> intracranial flow and <a style="color:green;">higer</a> pulsatility transmission/stiffness.
- [ ] Plaque burden may show strong associations with intracranial pulsatility transmission.
- [ ] Systemic vascular risk may act as a proximal driver of early cerebral vascular disfunction before araises cognitive sympthoms.

<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">
  <h2 style="color: #27ae60; border-bottom: 1px solid #27ae60; padding-bottom: 10px;">Previously reported:</h2>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Vascular abnormalies:</strong> 
    On at-risk macro-microvascular studies, cardiovascular risk factors are associated with later AD risk, possibly propossing vascular dysfunction as major co-driver of AD. <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6338083/">Vascular dysfunction – the disregarded partner of Alzheimer’s disease</a>] <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5318539/">Macrovascular and microvascular cerebral blood flow in adults at risk for Alzheimer's disease</a>]
  </div>
</div>

<a id="hyp_hemo_cog"></a>

### V. Hemodynamics vs. Cognition

---

Study the relationship between brain <b>4DFLow MRI</b> hemodynamics and <b>Cognitive Function</b> (neuropsycological battery).

Working hypothesis
:  Even in cognitively healthy middle-aged individuals, subtle hemodynamic abnormalities will be associated with worse cognitive performace.

Possible expected patterns:
- [ ] <a style="color:red;">Lower</a> flow and <a style="color:green;">higer</a> pulsatility/stiffness may relate to worse executive function, processing speed or memory.
- [ ] Hemodynamic markers may capture preclinical dysfunction prior clear clinical impairment emerges.

<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">
  <h2 style="color: #27ae60; border-bottom: 1px solid #27ae60; padding-bottom: 10px;">Previously reported:</h2>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Mild Cognitive Impairment:</strong> 
     On symptomatic-but-early MCI individuals, <a style="color:red;">reduced</a> ICAs & MCAs (Anterior circulation) mean flow possitively relates with <a style="color:red;">worse</a> executive performance. <br>
     Additionally, <a style="color:red;">lower</a> ICAs flow was associated with amyloid positivity. <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5617115/">Intracranial arterial 4D-flow in individuals with Mild Cognitive Impairment is associated with cognitive performance and amyloid positivity</a>]
  </div>
</div>

<a id="hyp_hemo_ad"></a>

### VI. Hemodynamics vs. Alzheimers Disease Biomarkers

---

Study the relationship between brain <b>4DFLow MRI</b>-derived hemodynamic phenotype and the biological signature of AD blood biomarkers.

Working hypothesis
:  A more adverse cerebral hemodynamic phenotype will be associated with a more abnormal biological profile related to preclinical Alzheimer’s disease.

Possible Expected patterns:
- [ ] <a style="color:green;">Lower</a> flow and <a style="color:red;">higer</a> pulsatility/stiffness may associate with more abnormal[/higer?] Amyloid-β [$Aβ_{42/40}$] and p-Tau [$p-Tau_{217}$] biomarkers.
- [ ] <b>Microglial Activation & Neurodegeneration:</b> Associations may also be seen with GFAP and Nfl (Neourofilaments), suggesting links with neuroinflammation and neuroaxonal injury.
- [ ] Effects may be stronger in <b>ApoE</b> risk carriers, suggesting a vascular-genetic interaction in preclinical AD.

<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">
  <h2 style="color: #27ae60; border-bottom: 1px solid #27ae60; padding-bottom: 10px;">Previously reported:</h2>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Aβ & p-Tau Biomarkers: </strong> 
     In Aβ & p-Tau biomarker-confirmed participants, including cognitively normal subjects, 4DFlow studies showed <a style="color:red;">shorter</a> transcapilary pulse-wave delay and <a style="color:red;">reduced</a> low frequency oscilators, interpreted as an <a style="color:green;">increased</a> stiffness and <a style="color:red;">reduced</a> vasomotion. Additionally, a significanlty higer pressence of microhemorrages was measured compared to controls. <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8719432/">Cerebrovascular stiffness and flow dynamics in the presence of amyloid and tau biomarkers</a>]
  </div>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">ApoE Carriers: </strong> 
     ApoE4 carriers show evidence of vascular dysfunction, greater Cerebral Amyloid Angiopathy (CAA), stronger anyloid effects than tau effects and more posterior white matter and posterior cortical abnormalies. While the pathomechanisms operating in non-carriers are less clear, the risk associated with APOE4 appears to be related to Aβ clearance through vasculature and TREM2-mediated microglial processes. <br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9266739/#sec5-ijms-23-07106">Vascular Dysfunction Is Central to Alzheimer’s Disease Pathogenesis in APOE e4 Carriers</a>]
  </div>
</div>

<a id="hyp_pred_long"></a>

### VII. Predictive value of 4DFLow-derived Markers in Longitudinal Changes

---

Long-term follow-up study of the brain hemodynamics for the prediction of AD and MCI risk using baseline hemodynamic parameters as predictors of the hemodynamic parameters evolution over time.

Working hypothesis
:  Baseline 4DFlow MRI hemodynamic markers may be used as predictors for future adverse brain and cognitive decline trajectories.

---

<a id="hyp_exp"></a>

### Exploratory hypotheses

Working hypothesis
:  Hemodynamic abnormalities in the vertebro-vasilar and posterior cerebral circulation may exibit stronger/earlyer associations with early AD-related markers than the anteriorcerebral circulation.

Working hypothesis
:  Purlatility transmition-based markers, such as damping indices or PITC will detect preclinical vascular dysfunctions earlyer than mean blood flow.

Working hypothesis
:  Abnormal arterial-to-venous pulsatility coupling may indentify an early failure, capturing a whole-brain vascular compliance rather than only arterial disease.

<div style="font-family: sans-serif; max-width: 900px; border-left: 4px solidrgb(8, 132, 41); padding-left: 20px; margin: 20px 0;">
  <h2 style="color: #27ae60; border-bottom: 1px solid #27ae60; padding-bottom: 10px;">Previously reported:</h2>
  <div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
    <span style="height: 10px; width: 10px; background-color: #27ae60; border-radius: 50%; display: inline-block; margin-right: 10px;"></span>
    <strong style="color: #27ae60;">Mild Cognitive Impairment:</strong> 
     AD studies found faster trnasmision of peak flow from the MCAs to the Superior Sagittal Sinus.<br>
     [Source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5464708/">Changes in intracranial venous blood flow and pulsatility in Alzheimer’s disease: A 4D flow MRI study</a>]
  </div>
</div>

Working hypothesis
:  Some individuals may preserver a 'normal' total intracranial flow while showing abnormal waveform shapes, delayed systolic timing, or increased pulsatility/stiffness.

Working hypothesis
:  Unsupervised clustering of multimodal vascular phenotipes (4DFlow, ASL, plaque/calcium burdens, blood pressure, cardiovascular risk factors, ApoE aplotype, etc...) may reveal disting preclinical phenotypes, interactions or subgroups enriched for combined Cardivascular and AD risk factors, rather than analyzing them separatedly.

---

<a id="papers"></a>

## Project Related Resources

<table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
  <thead>
    <tr style="background-color:rgb(90, 90, 90); border-bottom: 2px solid #ddd;">
      <th style="padding: 10px; text-align: left;">Type</th>
      <th style="padding: 10px; text-align: left;">Source</th>
      <th style="padding: 10px; text-align: left;">Overview</th>
      <th style="padding: 10px; text-align: left;">Key Insights</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="4" style="padding: 8px; border-bottom: 1px solid #eee;">PESA</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://www.sciencedirect.com/science/article/pii/S0002870324002618?via%3Dihub">Subclinical atherosclerosis and brain health in midlife: Rationale and design of the PESA-Brain study</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">PESA-Brain Study Description</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="4" style="padding: 8px; border-bottom: 1px solid #eee;">General</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6338083/">Vascular dysfunction – the disregarded partner of Alzheimer’s disease</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9266739/">Vascular Dysfunction Is Central to Alzheimer’s Disease Pathogenesis in APOE e4 Carriers</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4982069/">Pathophysiology of Hypoperfusion of the Precuneus in Early Alzheimer's Disease</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="4" style="padding: 8px; border-bottom: 1px solid #eee;">4DFlow MRI</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10140641/">Normative Cerebral Hemodynamics in Middle-aged and Older Adults Using 4D Flow MRI: Initial Analysis of Vascular Aging</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">CW & Venous MANUAL Vessel Segmentation</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5318539/">Macrovascular and microvascular cerebral blood flow in adults at risk for Alzheimer's disease</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">Micro-Macrovascular coupling</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://www.nature.com/articles/s41598-024-63312-4">Measuring global cerebrovascular pulsatility transmission using 4D flow MRI</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">QVT+ PITC</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8217890/">Cerebral arterial pulsatility is linked to hippocampal microvascular function and episodic memory in healthy older adults </a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">Pulsatility vs. WMH</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5617115/">Intracranial arterial 4D-flow in individuals with Mild Cognitive Impairment is associated with cognitive performance and amyloid positivity</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">MCI vs Flow</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7476069/">Intracranial vascular flow oscillations in Alzheimer’s disease from 4D flow MRI</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">Flow|PI vs AD</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8719432/">Cerebrovascular stiffness and flow dynamics in the presence of amyloid and tau biomarkers</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">Flow|PI vs AD</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5464708/">Changes in intracranial venous blood flow and pulsatility in Alzheimer’s disease: A 4D flow MRI study</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">Flow|PI vs AD</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC9266739/#sec5-ijms-23-07106">Vascular Dysfunction Is Central to Alzheimer’s Disease Pathogenesis in APOE e4 Carriers</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">ApoE e4 Carriers</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="4" style="padding: 8px; border-bottom: 1px solid #eee;">TOF MRI</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>📑</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://www.sciencedirect.com/science/article/pii/S1053811922005420">eICAB: A novel deep learning pipeline for Circle of Willis multiclass segmentation and analysis</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">CW & WB AUTO Vessel Segmentation</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
    <tr style="background-color: rgb(90, 90, 90); font-style: italic;">
      <td colspan="4" style="padding: 8px; border-bottom: 1px solid #eee;">Additional Media</td>
    </tr>
    <tr>
      <td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>🎬</b></td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">
        <a href="https://asl-docs.readthedocs.io/en/latest/analysis_guide.html">ASL Analysis Guide | BASIL Doc.</a>
      </td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
      <td style="padding: 8px; border-bottom: 1px solid #eee;">-</td>
    </tr>
  </tbody>
</table>

---

<a id="appendix"></a>

## Appendix 

<div style="display: flex; flex-direction: row; justify-content: center; gap: 20px; width: 100%; margin: 20px 0;">
  <div style="flex: 1; max-width: 45%; text-align: center; padding: 10px; border-radius: 8px;">
    <div class="figure-card">
      <div class="figure-title">Major Brain Regions</div>
      <svg viewBox="0 0 520 350" width="100%" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="230" cy="175" rx="170" ry="110" fill="#8fb9c9"/>
        <ellipse cx="170" cy="135" rx="85" ry="55" fill="#6fa8dc"/>
        <ellipse cx="255" cy="110" rx="65" ry="45" fill="#8e7cc3"/>
        <ellipse cx="315" cy="175" rx="110" ry="70" fill="#f6b26b"/>
        <ellipse cx="355" cy="140" rx="55" ry="48" fill="#e06666"/>
        <ellipse cx="355" cy="245" rx="55" ry="42" fill="#c9a66b"/>
        <rect x="230" y="220" width="20" height="70" fill="#f9cb9c"/>
        <text x="95" y="70" class="label-text">Frontal lobe</text>
        <text x="250" y="55" class="label-text">Parietal lobe</text>
        <text x="360" y="95" class="label-text">Occipital lobe</text>
        <text x="275" y="210" class="label-text">Temporal lobe</text>
        <text x="365" y="275" class="label-text">Cerebellum</text>
        <text x="255" y="310" class="label-text">Brainstem</text>
      </svg>
      <div class="figure-caption">
        Simplified lateral-view brain schematic showing the major macroscopic regions.
      </div>
    </div>
  </div>
  <div style="flex: 1; max-width: 45%; text-align: center; padding: 10px; border-radius: 8px;">
    <div class="figure-card">
      <div class="figure-title">Functional Organization</div>
      <svg viewBox="0 0 520 350" width="100%" xmlns="http://www.w3.org/2000/svg">
        <rect x="35" y="40" width="180" height="48" fill="#d9ead3" stroke="#b6d7a8"/>
        <text x="125" y="68" text-anchor="middle" class="label-text">Frontal: planning / control</text>
        <rect x="290" y="40" width="180" height="48" fill="#cfe2f3" stroke="#9fc5e8"/>
        <text x="380" y="68" text-anchor="middle" class="label-text">Parietal: integration / spatial</text>
        <rect x="35" y="130" width="180" height="48" fill="#f9cb9c" stroke="#f6b26b"/>
        <text x="125" y="158" text-anchor="middle" class="label-text">Temporal: memory / language</text>
        <rect x="290" y="130" width="180" height="48" fill="#f4cccc" stroke="#ea9999"/>
        <text x="380" y="158" text-anchor="middle" class="label-text">Occipital: vision</text>
        <rect x="35" y="220" width="180" height="48" fill="#ead1dc" stroke="#d5a6bd"/>
        <text x="125" y="248" text-anchor="middle" class="label-text">Cerebellum: coordination / timing</text>
        <rect x="290" y="220" width="180" height="48" fill="#fff2cc" stroke="#ffd966"/>
        <text x="380" y="248" text-anchor="middle" class="label-text">Brainstem: autonomic control</text>
      </svg>
      <div class="figure-caption">
        Summary of the main functional roles of the large-scale brain regions.
      </div>
    </div>
  </div>
</div>

<!-- ```{.panel-tabset} -->
<div class="figure-card">
  <div class="figure-title">Brain Anatomy and Functional Areas</div>
  <div class="figure-subtitle">Reference overview of major lobes, deep structures, and cortical functional regions.</div>
  <img src="./plots/report_appendix/licensed-image.jpg" style="width:100%; max-width:1000px; display:block; margin:auto;">
  <div class="figure-caption">
    Lobar organization and selected functional areas of the brain. For this report, the most relevant systems are frontal-parietal control regions, temporal memory-related structures, posterior cortical areas, cerebellum, and brainstem.
  </div>
</div>

<div class="figure-card">
  <div class="figure-title">Macroscopic Brain Vasculature</div>
  <div class="figure-subtitle">Reference view of the major arterial supply pathways and the Circle of Willis.</div>
  <img src="./plots/report_appendix/cow.png" style="width:100%; max-width:1000px; display:block; margin:auto;">
  <div class="figure-caption">
    The brain receives arterial inflow from the internal carotid and vertebro-basilar systems. The Circle of Willis provides the main intracranial collateral routing structure linking anterior and posterior circulation.
  </div>
</div>
<!-- ``` -->

<div class="figure-card">
  <div class="figure-title">Multiscale Cerebrovascular System</div>
  <div class="figure-subtitle">Systemic inflow, intracranial arterial routing, tissue perfusion, watershed vulnerability, venous drainage, and clearance.</div>
  <svg viewBox="0 0 1280 620" width="100%" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrowhead" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
        <polygon points="0 0, 10 4, 0 8" fill="#4b5563"/>
      </marker>
      <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#000000" flood-opacity="0.10"/>
      </filter>
      <style>
        .box {
          fill: #ffffff;
          stroke: #d6dbe1;
          stroke-width: 1.2;
          rx: 14;
          ry: 14;
        }
        .title {
          font-family: sans-serif;
          font-size: 16px;
          font-weight: 700;
          fill: #1f2937;
        }
        .label {
          font-family: sans-serif;
          font-size: 13px;
          fill: #1f2937;
        }
        .note {
          font-family: sans-serif;
          font-size: 11px;
          fill: #5b6573;
        }
        .section {
          font-family: sans-serif;
          font-size: 13px;
          font-weight: 700;
        }
        .arrow {
          stroke: #4b5563;
          stroke-width: 2.4;
          fill: none;
          marker-end: url(#arrowhead);
        }
        .soft-arrow {
          stroke: #7b8794;
          stroke-width: 1.8;
          fill: none;
          marker-end: url(#arrowhead);
        }
      </style>
    </defs>
    <!-- Section backgrounds -->
    <rect x="24" y="54" width="210" height="170" rx="20" fill="#fdf0f0" stroke="#f3c7c7"/>
    <rect x="260" y="54" width="360" height="170" rx="20" fill="#fdf0f0" stroke="#f3c7c7"/>
    <rect x="646" y="54" width="610" height="350" rx="20" fill="#eef9f1" stroke="#cfe8d6"/>
    <rect x="260" y="432" width="996" height="138" rx="20" fill="#f6effb" stroke="#decff0"/>
    <text x="42" y="82" class="section" fill="#b14b4b">Systemic inflow</text>
    <text x="278" y="82" class="section" fill="#b14b4b">Intracranial arterial routing</text>
    <text x="664" y="82" class="section" fill="#3d8156">Tissue-scale circulation</text>
    <text x="278" y="460" class="section" fill="#7e57a7">Clearance and fluid exchange</text>
    <!-- Systemic inflow -->
    <rect x="48" y="112" width="80" height="70" class="box" filter="url(#softShadow)"/>
    <text x="88" y="140" text-anchor="middle" class="title">Heart</text>
    <text x="88" y="160" text-anchor="middle" class="label">pulse source</text>
    <rect x="146" y="112" width="64" height="70" class="box" filter="url(#softShadow)"/>
    <text x="178" y="140" text-anchor="middle" class="title">MAP</text>
    <text x="178" y="160" text-anchor="middle" class="note">pressure load</text>
    <text x="48" y="205" class="note">Cardiac pulsation and systemic pressure generate</text>
    <text x="48" y="220" class="note">the inflow waveform received by the brain.</text>
    <!-- Arterial routing -->
    <rect x="294" y="112" width="130" height="70" class="box" filter="url(#softShadow)"/>
    <text x="359" y="138" text-anchor="middle" class="title">Large arteries</text>
    <text x="359" y="159" text-anchor="middle" class="label">ICA / Basilar</text>
    <rect x="460" y="112" width="130" height="70" class="box" filter="url(#softShadow)"/>
    <text x="525" y="138" text-anchor="middle" class="title">Circle of Willis</text>
    <text x="525" y="159" text-anchor="middle" class="label">collateral hub</text>
    <text x="295" y="205" class="note">4DFlow main domain: bulk flow, velocity, PI, PWV,</text>
    <text x="295" y="220" class="note">damping, transmission, and arterial asymmetry.</text>
    <!-- Territories -->
    <rect x="692" y="112" width="170" height="62" class="box" filter="url(#softShadow)"/>
    <text x="777" y="137" text-anchor="middle" class="title">Anterior territory</text>
    <text x="777" y="158" text-anchor="middle" class="label">ACA / MCA</text>
    <rect x="1030" y="112" width="180" height="62" class="box" filter="url(#softShadow)"/>
    <text x="1120" y="137" text-anchor="middle" class="title">Posterior territory</text>
    <text x="1120" y="158" text-anchor="middle" class="label">PCA / vertebro-basilar</text>
    <!-- Microvascular beds -->
    <rect x="678" y="224" width="180" height="64" class="box" filter="url(#softShadow)"/>
    <text x="768" y="249" text-anchor="middle" class="title">Anterior microvasculature</text>
    <text x="768" y="270" text-anchor="middle" class="label">arterioles + capillaries</text>
    <rect x="1032" y="224" width="180" height="64" class="box" filter="url(#softShadow)"/>
    <text x="1122" y="249" text-anchor="middle" class="title">Posterior microvasculature</text>
    <text x="1122" y="270" text-anchor="middle" class="label">arterioles + capillaries</text>
    <!-- Watershed -->
    <path d="M930 216 L978 242 L978 292 L930 318 L882 292 L882 242 Z"
          fill="#fff6c7" stroke="#e5c95b" stroke-width="1.5" filter="url(#softShadow)"/>
    <text x="930" y="250" text-anchor="middle" class="title" style="font-size:15px;">Watershed</text>
    <text x="930" y="270" text-anchor="middle" class="label">borderzone</text>
    <text x="930" y="289" text-anchor="middle" class="note">hypoperfusion-sensitive</text>
    <!-- Perfusion / tissue -->
    <ellipse cx="930" cy="366" rx="150" ry="72" fill="#dcecf8" stroke="#a9c4d6" stroke-width="1.5"/>
    <ellipse cx="880" cy="350" rx="46" ry="26" fill="#f6c5c5" opacity="0.95"/>
    <ellipse cx="978" cy="348" rx="42" ry="24" fill="#f8d9a8" opacity="0.95"/>
    <ellipse cx="892" cy="392" rx="50" ry="26" fill="#c8daf0" opacity="0.95"/>
    <ellipse cx="979" cy="392" rx="42" ry="24" fill="#d7efcf" opacity="0.95"/>
    <text x="930" y="364" text-anchor="middle" class="title">Brain tissue perfusion</text>
    <text x="930" y="385" text-anchor="middle" class="label">CBF / metabolic support</text>
    <text x="930" y="404" text-anchor="middle" class="note">gray matter + white matter</text>
    <!-- Venous -->
    <rect x="1098" y="330" width="130" height="72" class="box" filter="url(#softShadow)"/>
    <text x="1163" y="357" text-anchor="middle" class="title">Venous return</text>
    <text x="1163" y="378" text-anchor="middle" class="label">sinuses / jugular</text>
    <!-- Clearance -->
    <rect x="430" y="486" width="150" height="52" class="box" filter="url(#softShadow)"/>
    <text x="505" y="508" text-anchor="middle" class="title">CSF</text>
    <text x="505" y="526" text-anchor="middle" class="label">fluid compartment</text>
    <rect x="620" y="486" width="170" height="52" class="box" filter="url(#softShadow)"/>
    <text x="705" y="508" text-anchor="middle" class="title">Perivascular spaces</text>
    <text x="705" y="526" text-anchor="middle" class="label">exchange pathways</text>
    <rect x="834" y="486" width="160" height="52" class="box" filter="url(#softShadow)"/>
    <text x="914" y="508" text-anchor="middle" class="title">Glymphatic flow</text>
    <text x="914" y="526" text-anchor="middle" class="label">waste clearance</text>
    <rect x="1032" y="486" width="170" height="52" class="box" filter="url(#softShadow)"/>
    <text x="1117" y="508" text-anchor="middle" class="title">Compliance / pressure</text>
    <text x="1117" y="526" text-anchor="middle" class="label">system-level coupling</text>
    <!-- Main arrows -->
    <path d="M210 147 C245 147, 260 147, 294 147" class="arrow"/>
    <path d="M424 147 C442 147, 448 147, 460 147" class="arrow"/>
    <path d="M590 147 C630 147, 650 143, 692 143" class="arrow"/>
    <path d="M590 147 C740 155, 920 150, 1030 143" class="arrow"/>
    <path d="M777 174 C777 188, 772 200, 768 224" class="arrow"/>
    <path d="M1120 174 C1120 188, 1122 202, 1122 224" class="arrow"/>
    <!-- Territory to watershed -->
    <path d="M858 255 C870 255, 876 255, 882 255" class="soft-arrow"/>
    <path d="M1032 255 C1014 255, 998 255, 978 255" class="soft-arrow"/>
    <!-- Microvasculature to tissue -->
    <path d="M768 288 C782 316, 820 332, 860 344" class="arrow"/>
    <path d="M1122 288 C1098 316, 1048 332, 1002 344" class="arrow"/>
    <path d="M930 318 C930 330, 930 338, 930 344" class="arrow"/>
    <!-- Perfusion to venous -->
    <path d="M1080 366 C1090 366, 1092 366, 1098 366" class="arrow"/>
    <!-- Perfusion / venous to clearance -->
    <path d="M845 428 C820 450, 770 470, 705 486" class="arrow"/>
    <path d="M930 438 C930 456, 922 470, 914 486" class="arrow"/>
    <path d="M1128 402 C1128 434, 1122 454, 1117 486" class="arrow"/>
    <!-- Clearance chain -->
    <path d="M580 512 C594 512, 603 512, 620 512" class="arrow"/>
    <path d="M790 512 C804 512, 814 512, 834 512" class="arrow"/>
    <path d="M994 512 C1008 512, 1016 512, 1032 512" class="arrow"/>
    <!-- Notes -->
    <text x="720" y="302" class="note">territory-specific microvascular beds</text>
    <text x="870" y="332" class="note">ASL main domain: tissue-level perfusion</text>
    <text x="1086" y="420" class="note">venous pulsatility and outflow reflect whole-system buffering</text>
    <text x="444" y="556" class="note">clearance, compliance, and waste handling are linked to vascular pulsatility and pressure propagation</text>
    <!-- Legend -->
    <circle cx="48" cy="594" r="6" fill="#e76f51"/>
    <text x="61" y="598" class="note">arterial transport</text>
    <circle cx="180" cy="594" r="6" fill="#52be80"/>
    <text x="193" y="598" class="note">microvascular / tissue perfusion</text>
    <circle cx="385" cy="594" r="6" fill="#5dade2"/>
    <text x="398" y="598" class="note">venous drainage</text>
    <circle cx="515" cy="594" r="6" fill="#af7ac5"/>
    <text x="528" y="598" class="note">CSF / clearance coupling</text>
    <circle cx="695" cy="594" r="6" fill="#f4d03f"/>
    <text x="708" y="598" class="note">watershed vulnerability</text>
  </svg>
  <div class="figure-caption">
    Schematic summary of the cerebrovascular system across scales. Systemic pulsatile inflow reaches the intracranial arterial tree, is redistributed through the Circle of Willis into anterior and posterior territories, and is then transmitted into territory-specific microvascular beds. Borderzone watershed regions are especially vulnerable to reduced perfusion or impaired buffering. Tissue perfusion, venous drainage, and CSF/perivascular clearance should therefore be interpreted as coupled properties of the same system.
  </div>
</div>
