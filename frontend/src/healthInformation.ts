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
    meaning: 'Your age is one part of the information used in this assessment.',
    relevance: 'Age is one part of a person’s overall heart-health information.',
    limitations: 'Age alone cannot describe your health or determine your personal chance of disease.',
    sources: [uci, who],
  },
  sex: {
    label: 'Recorded sex category',
    categories: { 0: 'Female', 1: 'Male' },
    meaning: 'The older study recorded this answer in two groups: female or male.',
    relevance: 'This answer is included because it appeared in the older records used by the computer.',
    limitations: 'The older records offered only two choices here. This does not fully describe a person’s sex, gender, or health.',
    sources: [uci, who],
  },
  cp: {
    label: 'Chest-pain category',
    categories: {
      0: 'Typical chest-pain pattern',
      1: 'Less typical chest-pain pattern',
      2: 'Other chest-pain pattern (non-anginal)',
      3: 'No chest-pain symptoms recorded',
    },
    meaning: 'One of four labels from the older study used to describe chest-pain history.',
    relevance: 'Chest discomfort can have many causes. A healthcare professional considers it with a person’s history and examination.',
    limitations: 'This older study category is not a diagnosis and does not explain the cause of any current symptoms.',
    sources: [uci, chestPain],
  },
  trestbps: {
    label: 'Blood pressure reading',
    unit: 'mmHg',
    meaning: 'Blood pressure describes how strongly blood presses against blood vessels. The older study includes one reading taken at rest.',
    relevance: 'A healthcare professional reviews blood pressure with other readings and health information.',
    limitations: 'This field does not include both numbers of a usual blood pressure reading, and one value cannot establish whether blood pressure is high or low.',
    sources: [uci, bloodPressure],
  },
  chol: {
    label: 'Total cholesterol recorded',
    unit: 'mg/dL',
    meaning: 'Cholesterol is a fat-like substance in the blood. The older study includes one total cholesterol reading.',
    relevance: 'A healthcare professional reviews cholesterol with other blood-fat readings and a person’s overall health.',
    limitations: 'This is one cholesterol reading. It does not show all types of cholesterol or establish a diagnosis.',
    sources: [uci, cholesterol],
  },
  fbs: {
    label: 'Blood sugar before eating',
    categories: {
      0: '120 mg/dL or lower',
      1: 'Above 120 mg/dL',
    },
    meaning: 'This answer records whether blood sugar before eating was above the level used in the older study: 120 mg/dL.',
    relevance: 'A blood test can measure sugar in your blood. The result depends on when you last ate and how the test was done.',
    limitations: 'This is a yes-or-no category from an older study, not a full blood-sugar reading or a diagnosis.',
    sources: [uci, glucose],
  },
  restecg: {
    label: 'Heart test at rest',
    categories: {
      0: 'Normal heart-test category recorded in the study',
      1: 'Heart-test electrical change category',
      2: 'Possible heart-muscle enlargement category',
    },
    meaning: 'An electrocardiogram (ECG) records the heart’s electrical signals. These are grouped answers from an older study.',
    relevance: 'A healthcare professional reviews an ECG with your symptoms and other test results.',
    limitations: 'This category cannot be interpreted by itself. A healthcare professional needs the original test and your health context.',
    sources: [uci, ecg],
  },
  thalach: {
    label: 'Highest heartbeat during activity',
    unit: 'beats per minute',
    meaning: 'The highest heart rate recorded during the exercise test.',
    relevance: 'An exercise test observes the heart while it works harder.',
    limitations: 'The meaning depends on the test protocol and personal context; this value is not a stand-alone health assessment.',
    sources: [uci, stressTest],
  },
  exang: {
    label: 'Chest discomfort during exercise',
    categories: { 0: 'No', 1: 'Yes' },
    meaning: 'A yes-or-no answer about whether chest discomfort was recorded during exercise.',
    relevance: 'Chest discomfort can have many causes and should be considered with symptoms and other health information.',
    limitations: 'This answer cannot tell why discomfort occurred. New or severe symptoms need prompt medical attention.',
    sources: [uci, chestPain, stressTest],
  },
  oldpeak: {
    label: 'Change in a heart-test reading during exercise',
    meaning: 'This number comes from a heart test during exercise. It compares a heart-test reading during activity with one taken at rest.',
    relevance: 'The heart test can show changes while the heart is working harder.',
    limitations: 'The older study does not give a unit for this number. A healthcare professional needs the original report to understand it.',
    sources: [uci, ecg, stressTest],
  },
  slope: {
    label: 'Heart-test pattern during exercise',
    categories: {
      0: 'Rising heart-test pattern',
      1: 'Flat heart-test pattern',
      2: 'Falling heart-test pattern',
    },
    meaning: 'This category describes whether a line on an ECG (electrocardiogram) was rising, flat, or falling during exercise.',
    relevance: 'The full heart test and a person’s health information are needed to understand this pattern.',
    limitations: 'This category cannot confirm a heart condition. A healthcare professional needs the original test and its context.',
    sources: [uci, ecg, stressTest],
  },
  ca: {
    label: 'Blood vessels counted in the test',
    categories: { 0: '0', 1: '1', 2: '2', 3: '3' },
    meaning: 'The older study recorded how many large blood vessels were seen during a special X-ray test.',
    relevance: 'Imaging tests can help healthcare professionals look at blood vessels and blood flow.',
    limitations: 'This number does not by itself mean that a vessel is blocked or establish a diagnosis.',
    sources: [uci, angiography],
  },
  thal: {
    label: 'Specialized heart-test category',
    categories: {
      1: 'Fixed test-finding category',
      2: 'Normal category recorded in the study',
      3: 'Reversible test-finding category',
    },
    meaning: 'One of the categories from a specialized heart test in the older study.',
    relevance: 'A healthcare professional needs the full report to understand this specialized test result.',
    limitations: 'These labels come from an older study. “Normal” here does not describe your overall health.',
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
