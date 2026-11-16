export type FeatureKey =
  | 'age'
  | 'sex'
  | 'cp'
  | 'trestbps'
  | 'chol'
  | 'fbs'
  | 'restecg'
  | 'thalach'
  | 'exang'
  | 'oldpeak'
  | 'slope'
  | 'ca'
  | 'thal';

type Source = { label: string; url: string; kind: 'dataset' | 'health' };
type FeatureInformation = {
  label: string;
  unit?: string;
  categories?: Record<number, string>;
  meaning: string;
  relevance: string;
  limitations: string;
  sources: Source[];
};

const uci: Source = {
  label: 'UCI Heart Disease dataset documentation',
  url: 'https://archive.ics.uci.edu/dataset/45/heart+disease',
  kind: 'dataset',
};
const who: Source = {
  label: 'World Health Organization: Cardiovascular diseases',
  url: 'https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)',
  kind: 'health',
};
const bloodPressure: Source = {
  label: 'MedlinePlus: High blood pressure',
  url: 'https://medlineplus.gov/highbloodpressure.html',
  kind: 'health',
};
const cholesterol: Source = {
  label: 'MedlinePlus: Cholesterol',
  url: 'https://medlineplus.gov/cholesterol.html',
  kind: 'health',
};
const glucose: Source = {
  label: 'MedlinePlus: Blood glucose test',
  url: 'https://medlineplus.gov/lab-tests/blood-glucose-test/',
  kind: 'health',
};
const ecg: Source = {
  label: 'MedlinePlus: Electrocardiogram',
  url: 'https://medlineplus.gov/lab-tests/electrocardiogram/',
  kind: 'health',
};
const stressTest: Source = {
  label: 'MedlinePlus: Stress tests',
  url: 'https://medlineplus.gov/lab-tests/stress-tests/',
  kind: 'health',
};
const chestPain: Source = {
  label: 'NHS: Chest pain',
  url: 'https://www.nhs.uk/symptoms/chest-pain/',
  kind: 'health',
};
const angiography: Source = {
  label: 'NHS: Angiography',
  url: 'https://www.nhs.uk/tests-and-treatments/angiography/',
  kind: 'health',
};
const aiGovernance: Source = {
  label: 'WHO: Ethics and governance of AI for health',
  url: 'https://www.who.int/publications/i/item/9789240029200',
  kind: 'health',
};

export const AI_HEALTH_SOURCE = aiGovernance;

export const healthInformation: Record<FeatureKey, FeatureInformation> = {
  age: {
    label: 'Your age',
    unit: 'years',
    meaning: 'Your age is one of the details used by the research model.',
    relevance: 'Age is one part of general cardiovascular health information.',
    limitations: 'Age alone cannot describe your health or determine your personal chance of disease.',
    sources: [uci, who],
  },
  sex: {
    label: 'Recorded sex category',
    categories: { 0: 'Female', 1: 'Male' },
    meaning: 'The original dataset records this field using two categories: female or male.',
    relevance: 'This is a demographic field the model learned from in its historical data.',
    limitations: 'This old dataset field is limited and is not a full description of sex, gender, or an individual’s health.',
    sources: [uci, who],
  },
  cp: {
    label: 'Chest-pain category',
    categories: {
      0: 'Typical angina category',
      1: 'Atypical angina category',
      2: 'Non-anginal pain category',
      3: 'Asymptomatic category',
    },
    meaning: 'A category recorded in the source dataset to describe chest-pain history.',
    relevance: 'Chest discomfort can have many causes and is considered alongside a person’s history and examination.',
    limitations: 'This historical category is not a diagnosis and does not determine the cause of any current symptoms.',
    sources: [uci, chestPain],
  },
  trestbps: {
    label: 'Blood pressure reading',
    unit: 'mmHg',
    meaning: 'Blood pressure describes the force of blood against artery walls. This dataset field is one resting reading.',
    relevance: 'A clinician considers blood pressure readings together with other readings and health information.',
    limitations: 'This field does not include both numbers of a usual blood pressure reading, and one value cannot establish whether blood pressure is high or low.',
    sources: [uci, bloodPressure],
  },
  chol: {
    label: 'Total cholesterol recorded',
    unit: 'mg/dL',
    meaning: 'Cholesterol is a fat-like substance in the blood. The dataset records a serum cholesterol value.',
    relevance: 'Cholesterol information is interpreted with other blood fats and a person’s overall health context.',
    limitations: 'This dataset value does not provide a complete modern cholesterol panel or establish a diagnosis.',
    sources: [uci, cholesterol],
  },
  fbs: {
    label: 'Blood sugar before eating',
    categories: {
      0: '120 mg/dL or lower',
      1: 'Above 120 mg/dL',
    },
    meaning: 'This historical field records whether fasting blood sugar was above 120 mg/dL in the source dataset.',
    relevance: 'Blood glucose tests measure sugar in the blood; clinicians interpret values alongside the time of eating, the test method, and a person’s health context.',
    limitations: 'This value is a simple category from an older dataset, not a full glucose reading or a diagnosis of diabetes or other blood-sugar problems.',
    sources: [uci, glucose],
  },
  restecg: {
    label: 'Heart test at rest',
    categories: {
      0: 'Normal ECG category',
      1: 'ST-T wave change category',
      2: 'Possible or definite left-ventricular enlargement category',
    },
    meaning: 'An ECG records the electrical signals made by the heart. These are grouped categories from the historical dataset.',
    relevance: 'ECG findings may help a clinician assess heart rhythm and electrical activity as part of a wider evaluation.',
    limitations: 'A category from this dataset is not a stand-alone diagnosis; a clinician needs the original tracing and context.',
    sources: [uci, ecg],
  },
  thalach: {
    label: 'Highest heartbeat during activity',
    unit: 'beats per minute',
    meaning: 'The highest heart rate recorded during the exercise test.',
    relevance: 'An exercise stress test observes the heart while it works harder.',
    limitations: 'The meaning depends on the test protocol and personal context; this value is not a stand-alone health assessment.',
    sources: [uci, stressTest],
  },
  exang: {
    label: 'Chest discomfort during exercise',
    categories: { 0: 'No', 1: 'Yes' },
    meaning: 'A yes/no category for exercise-related chest discomfort in the source data.',
    relevance: 'Chest discomfort can have many causes and should be considered with symptoms and other clinical information.',
    limitations: 'This dataset field cannot determine why discomfort occurred. New or severe symptoms need prompt medical attention.',
    sources: [uci, chestPain, stressTest],
  },
  oldpeak: {
    label: 'Change in a heart-test reading during exercise',
    meaning: 'A numerical ECG-related reading recorded during exercise compared with rest.',
    relevance: 'Exercise ECG recordings can show changes while the heart is working harder.',
    limitations: 'The dataset does not specify a unit for this field. Its clinical meaning requires the original test and professional interpretation.',
    sources: [uci, ecg, stressTest],
  },
  slope: {
    label: 'Heart-test pattern during exercise',
    categories: {
      0: 'Upsloping category',
      1: 'Flat category',
      2: 'Downsloping category',
    },
    meaning: 'A category describing the slope of a heart-test segment at peak exercise.',
    relevance: 'Exercise ECG patterns are interpreted by clinicians with the full test and a person’s history.',
    limitations: 'This category cannot independently establish a heart condition. The original ECG and test context are needed.',
    sources: [uci, ecg, stressTest],
  },
  ca: {
    label: 'Blood vessels counted in the test',
    categories: { 0: '0', 1: '1', 2: '2', 3: '3' },
    meaning: 'The dataset records how many major vessels were seen during a fluoroscopy-based test.',
    relevance: 'Imaging tests can help clinicians examine blood vessels and blood flow.',
    limitations: 'This number does not by itself mean that a vessel is blocked or establish a diagnosis.',
    sources: [uci, angiography],
  },
  thal: {
    label: 'Specialized heart-test category',
    categories: {
      1: 'Fixed-defect category',
      2: 'Normal category in the dataset',
      3: 'Reversible-defect category',
    },
    meaning: 'A specialized test category recorded in this historical dataset.',
    relevance: 'Stress-test results are interpreted by a healthcare professional using the full report.',
    limitations: 'The source field name and categories come from an older dataset. Do not treat “normal” here as a statement about your overall health.',
    sources: [uci, stressTest],
  },
};

export const featureKeys: FeatureKey[] = [
  'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
  'exang', 'oldpeak', 'slope', 'ca', 'thal',
];

export const categoryOptions: Partial<Record<FeatureKey, { value: number; label: string }[]>> = {};
for (const feature of featureKeys) {
  const categories = healthInformation[feature].categories;
  if (categories) {
    categoryOptions[feature] = Object.entries(categories).map(([value, label]) => ({
      value: Number(value),
      label,
    }));
  }
}

export function formatFeatureValue(feature: FeatureKey, value: number): string {
  const category = healthInformation[feature].categories?.[value];
  if (category) return category;
  const displayValue = feature === 'oldpeak' ? value.toFixed(1) : String(value);
  const unit = healthInformation[feature].unit;
  return unit ? `${displayValue} ${unit}` : displayValue;
}

export const classificationThreshold = 0.5;
