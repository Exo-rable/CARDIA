import { categoryOptions, featureKeys, healthInformation, type FeatureKey } from './healthInformation';

export type Patient = {
  age:number;
  sex:number;
  cp:number;
  trestbps:number;
  chol:number;
  fbs:number;
  restecg:number;
  thalach:number;
  exang:number;
  oldpeak:number;
  slope:number;
  ca:number;
  thal:number;
};
export type PatientForm=Partial<Record<FeatureKey,number>>;

export type Explanation = {
  output_space:'log_odds';
  base_value:number;
  contributions:Array<{feature:FeatureKey;value:number}>;
  note:string;
};

export type InputRule = {
  type:'integer'|'number';
  minimum?:number;
  maximum?:number;
  enum?:number[];
};

export type InputSchema = Record<FeatureKey,InputRule>;

export type ModelInfo = {
  selected_model:string;
  n_train?:number;
  n_test?:number;
  test_metrics:Record<string,unknown>;
  cv_table:Array<Record<string,unknown>>;
  versions?:Record<string,unknown>;
  warning?:string;
};

export type Prediction = {
  probability:number;
  model_name:string;
  model_version:string;
  disclaimer:string;
};

export type AssessmentRecord = {
  schemaVersion:2;
  id:string;
  userId:string;
  createdAt:string;
  inputs:Patient|null;
  modelName:string;
  modelVersion:string;
  predictedClass:0|1;
  probabilities:{class0:number;class1:number};
  decisionThreshold:number;
  explanation:Explanation|null;
  explanationError?:string;
  legacyAge?:number;
};

export type DemoState = {
  userId:string;
  profileName:string;
  assessments:AssessmentRecord[];
  warning:string;
};

const HISTORY_KEY='cardia.demoHistory';
const PROFILE_ID_KEY='cardia.profileId';
const PROFILE_NAME_KEY='cardia.profileName';

export const initialPatient:Patient = {
  age:55,sex:1,cp:1,trestbps:130,chol:240,fbs:0,restecg:1,
  thalach:152,exang:0,oldpeak:0.8,slope:1,ca:0,thal:2,
};

export function isPatient(value:unknown):value is Patient {
  if(typeof value!=='object'||value===null)return false;
  const data=value as Record<string,unknown>;
  return Object.keys(data).length===featureKeys.length
    &&featureKeys.every(key=>typeof data[key]==='number'&&Number.isFinite(data[key]));
}

export function validatePatient(value:PatientForm,schema:InputSchema,keys:FeatureKey[]=featureKeys):string[] {
  const errors:string[]=[];
  if(!featureKeys.every(key=>key in schema))return ['The service did not provide validation rules for all required answers.'];
  for(const key of keys){
    const rule=schema[key];
    const fieldValue=value[key];
    const label=healthInformation[key].label;
    if(typeof fieldValue!=='number'||!Number.isFinite(fieldValue)){
      errors.push(`${label} is required.`);
      continue;
    }
    if(rule.enum&&!rule.enum.includes(fieldValue)){
      errors.push(`Choose an available option for ${label}.`);
    }else if(rule.type==='integer'&&!Number.isInteger(fieldValue)){
      errors.push(`${label} must be a whole number.`);
    }else if(rule.minimum!==undefined&&fieldValue<rule.minimum){
      errors.push(`${label} must be at least ${rule.minimum}.`);
    }else if(rule.maximum!==undefined&&fieldValue>rule.maximum){
      errors.push(`${label} must be no more than ${rule.maximum}.`);
    }
  }
  return errors;
}

export function isInputSchema(value:unknown):value is InputSchema {
  if(typeof value!=='object'||value===null)return false;
  const data=value as Record<string,unknown>;
  return Object.keys(data).length===featureKeys.length&&featureKeys.every(key=>{
    const rule=data[key];
    if(typeof rule!=='object'||rule===null)return false;
    const item=rule as Record<string,unknown>;
    if(item.type!=='integer'&&item.type!=='number')return false;
    if(item.enum!==undefined&&(!Array.isArray(item.enum)||!item.enum.every(entry=>typeof entry==='number')))return false;
    if(item.minimum!==undefined&&typeof item.minimum!=='number')return false;
    if(item.maximum!==undefined&&typeof item.maximum!=='number')return false;
    const options=categoryOptions[key];
    if(Array.isArray(item.enum)){
      return item.enum.length>0&&options?.length===item.enum.length
        &&item.enum.every(option=>options.some(item=>item.value===option));
    }
    return typeof item.minimum==='number'&&typeof item.maximum==='number';
  });
}

export function isPrediction(value:unknown):value is Prediction {
  if(typeof value!=='object'||value===null)return false;
  const data=value as Record<string,unknown>;
  return typeof data.probability==='number'
    &&Number.isFinite(data.probability)
    &&data.probability>=0
    &&data.probability<=1
    &&typeof data.model_name==='string'
    &&typeof data.model_version==='string'
    &&typeof data.disclaimer==='string';
}

export function isExplanation(value:unknown):value is Explanation {
  if(typeof value!=='object'||value===null)return false;
  const data=value as Record<string,unknown>;
  if(data.output_space!=='log_odds'||typeof data.base_value!=='number'||!Number.isFinite(data.base_value)||typeof data.note!=='string'||!Array.isArray(data.contributions))return false;
  const seen=new Set<FeatureKey>();
  for(const contribution of data.contributions){
    if(typeof contribution!=='object'||contribution===null)return false;
    const item=contribution as Record<string,unknown>;
    const feature=featureKeys.find(key=>key===item.feature);
    if(!feature||typeof item.value!=='number'||!Number.isFinite(item.value)||seen.has(feature))return false;
    seen.add(feature);
  }
  return seen.size===featureKeys.length;
}

function isRecord(value:unknown):value is Record<string,unknown> {
  return typeof value==='object'&&value!==null;
}

function savedAssessment(value:unknown,userId:string):AssessmentRecord|null {
  if(!isRecord(value)||typeof value.id!=='string')return null;
  if(value.schemaVersion===2){
    if(typeof value.userId!=='string'||typeof value.createdAt!=='string'||typeof value.modelName!=='string'||typeof value.modelVersion!=='string')return null;
    if(!Number.isFinite(Date.parse(value.createdAt)))return null;
    if(value.inputs!==null&&!isPatient(value.inputs))return null;
    if(value.inputs===null&&value.explanation!==null)return null;
    if(value.predictedClass!==0&&value.predictedClass!==1)return null;
    if(typeof value.decisionThreshold!=='number'||!Number.isFinite(value.decisionThreshold)||value.decisionThreshold<0||value.decisionThreshold>1)return null;
    if(!isRecord(value.probabilities)
      ||typeof value.probabilities.class0!=='number'
      ||typeof value.probabilities.class1!=='number'
      ||!Number.isFinite(value.probabilities.class0)
      ||!Number.isFinite(value.probabilities.class1)
      ||value.probabilities.class0<0||value.probabilities.class0>1
      ||value.probabilities.class1<0||value.probabilities.class1>1)return null;
    if(Math.abs(value.probabilities.class0+value.probabilities.class1-1)>1e-8)return null;
    if(value.predictedClass!==(value.probabilities.class1>=value.decisionThreshold?1:0))return null;
    if(value.explanation!==null&&!isExplanation(value.explanation))return null;
    const probabilities={
      class0:value.probabilities.class0,
      class1:value.probabilities.class1,
    };
    return {
      schemaVersion:2,
      id:value.id,
      userId:value.userId,
      createdAt:value.createdAt,
      inputs:value.inputs,
      modelName:value.modelName,
      modelVersion:value.modelVersion,
      predictedClass:value.predictedClass,
      probabilities,
      decisionThreshold:value.decisionThreshold,
      explanation:value.explanation,
      ...(typeof value.explanationError==='string'?{explanationError:value.explanationError}:{}),
      ...(typeof value.legacyAge==='number'?{legacyAge:value.legacyAge}:{}),
    };
  }
  if(typeof value.date==='string'&&typeof value.probability==='number'
    &&Number.isFinite(value.probability)&&value.probability>=0&&value.probability<=1
    &&typeof value.model==='string'&&typeof value.age==='number'){
    const predictedClass=value.probability>=0.5?1:0;
    return {
      schemaVersion:2,
      id:value.id,
      userId,
      createdAt:value.date,
      inputs:null,
      modelName:value.model,
      modelVersion:'unknown',
      predictedClass,
      probabilities:{class0:1-value.probability,class1:value.probability},
      decisionThreshold:0.5,
      explanation:null,
      explanationError:'Full measurements and explanation were not saved with this older demo record.',
      legacyAge:value.age,
    };
  }
  return null;
}

export function loadDemoState():DemoState {
  const warnings:string[]=[];
  let userId='local-demo-unavailable';
  let profileName='';
  let assessments:AssessmentRecord[]=[];
  try{
    const savedId=localStorage.getItem(PROFILE_ID_KEY);
    userId=savedId||crypto.randomUUID();
    if(!savedId)localStorage.setItem(PROFILE_ID_KEY,userId);
    profileName=localStorage.getItem(PROFILE_NAME_KEY)||'';
    const stored=localStorage.getItem(HISTORY_KEY);
    if(stored){
      const parsed:unknown=JSON.parse(stored);
      if(!Array.isArray(parsed))throw new Error('Saved assessment history is not a list.');
      const records=parsed.map(item=>savedAssessment(item,userId));
      assessments=records.filter((item):item is AssessmentRecord=>item!==null);
      if(assessments.length!==parsed.length){
        warnings.push('Some unreadable local demo records were skipped.');
      }
      if(parsed.some(item=>isRecord(item)&&item.schemaVersion!==2)){
        localStorage.setItem(HISTORY_KEY,JSON.stringify(assessments));
      }
    }
  }catch(error){
    warnings.push(`Browser demo data could not be fully loaded: ${error instanceof Error?error.message:'storage access failed'}`);
  }
  return {userId,profileName,assessments,warning:warnings.join(' ')};
}

export function saveDemoAssessments(assessments:AssessmentRecord[]):void {
  localStorage.setItem(HISTORY_KEY,JSON.stringify(assessments));
}

export function saveDemoProfileName(name:string):void {
  localStorage.setItem(PROFILE_NAME_KEY,name);
}

export function clearDemoAssessments():void {
  localStorage.removeItem(HISTORY_KEY);
}
