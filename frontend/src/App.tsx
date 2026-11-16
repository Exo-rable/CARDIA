import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Activity, ArrowLeft, ArrowRight, BookOpen, CalendarDays, Check, ChevronDown,
  CircleHelp, ClipboardList, Download, HeartPulse, History, Home, LoaderCircle,
  Info, Menu, RotateCcw, ShieldCheck, Stethoscope, Trash2, X,
} from 'lucide-react';
import './patient-friendly.css';
import {
  isExplanation, isInputSchema, isPatient, isPrediction,
  loadDemoState, saveDemoAssessments, saveDemoProfileName, clearDemoAssessments,
  validatePatient,
  type AssessmentRecord, type Explanation, type InputSchema, type ModelInfo,
  type Patient, type PatientForm,
} from './assessment';
import {
  categoryOptions, classificationThreshold, featureKeys, formatFeatureValue,
  healthInformation, AI_HEALTH_SOURCE, type FeatureKey,
} from './healthInformation';
import type { NumericFeature } from './HistoryChart';

type Page = 'home'|'assessment'|'history'|'learning'|'about';
type Step = { title:string; description:string; keys:FeatureKey[] };

const API=(import.meta.env.VITE_API_URL??(import.meta.env.DEV?'http://localhost:8000':'')).replace(/\/+$/,'');
const HistoryChart=lazy(()=>import('./HistoryChart'));
const steps:Step[]=[
  {title:'About You',description:'Enter basic information from your records.',keys:['age','sex']},
  {title:'Health Measurements',description:'Use readings as they appear in your records.',keys:['trestbps','chol','fbs']},
  {title:'Heart and Activity Information',description:'Use the exact values and categories shown in a medical report. Do not guess.',keys:['cp','restecg','exang','thalach','oldpeak','slope','ca','thal']},
  {title:'Review and Submit',description:'Check your answers and edit anything that needs correcting.',keys:[]},
];
const labels:Record<FeatureKey,string>={
  age:healthInformation.age.label,sex:healthInformation.sex.label,cp:healthInformation.cp.label,
  trestbps:healthInformation.trestbps.label,chol:healthInformation.chol.label,fbs:healthInformation.fbs.label,
  restecg:healthInformation.restecg.label,thalach:healthInformation.thalach.label,exang:healthInformation.exang.label,
  oldpeak:healthInformation.oldpeak.label,slope:healthInformation.slope.label,ca:healthInformation.ca.label,
  thal:healthInformation.thal.label,
};
const navItems:{id:Page;label:string;icon:typeof Home}[]=[
  {id:'home',label:'Home',icon:Home},
  {id:'assessment',label:'Heart health check',icon:ClipboardList},
  {id:'history',label:'Assessment history',icon:History},
  {id:'learning',label:'Learn',icon:BookOpen},
  {id:'about',label:'About CARDIA',icon:CircleHelp},
];

function isModelInfo(value:unknown):value is ModelInfo {
  if(typeof value!=='object'||value===null)return false;
  const data=value as Record<string,unknown>;
  return typeof data.selected_model==='string'
    &&typeof data.test_metrics==='object'&&data.test_metrics!==null
    &&Array.isArray(data.cv_table);
}

async function apiRequest(path:string,init?:RequestInit):Promise<unknown> {
  if(import.meta.env.PROD){
    const serviceUrl=new URL(API||window.location.origin,window.location.origin);
    const isLocal=(hostname:string)=>hostname==='localhost'||hostname==='127.0.0.1'||hostname==='[::1]';
    if((!isLocal(window.location.hostname)&&window.location.protocol!=='https:')
      ||(!isLocal(serviceUrl.hostname)&&serviceUrl.protocol!=='https:')){
      throw new Error('CARDIA must use HTTPS in production before connecting to the service.');
    }
  }
  let response:Response;
  try{
    response=await fetch(`${API}${path}`,init);
  }catch{
    throw new Error(`CARDIA could not connect. Check that the service is running, then try again.`);
  }
  const body:unknown=await response.json().catch(()=>null);
  if(!response.ok){
    const detail=typeof body==='object'&&body!==null&&'detail'in body
      ?String((body as Record<string,unknown>).detail)
      :`The CARDIA service returned an error (${response.status}).`;
    throw new Error(detail);
  }
  if(body===null)throw new Error('CARDIA received a response it could not read. Please try again.');
  return body;
}

function dateLabel(value:string,includeTime=false):string {
  const date=new Date(value);
  if(!Number.isFinite(date.getTime()))return 'Date unavailable';
  return date.toLocaleString(undefined,includeTime
    ?{dateStyle:'medium',timeStyle:'short'}
    :{dateStyle:'medium'});
}

function patternLabel(record:AssessmentRecord):string {
  return record.predictedClass===1
    ?'The AI found a pattern linked to heart-disease records in the study.'
    :'The AI did not find that study pattern in the information entered.';
}

function Field({
  name,value,schema,onChange,onOpenInfo,
}:{
  name:FeatureKey;
  value:number|undefined;
  schema:InputSchema;
  onChange:(value:number|undefined)=>void;
  onOpenInfo:(feature:FeatureKey)=>void;
}) {
  const rule=schema[name];
  const info=healthInformation[name];
  const options=categoryOptions[name]||[];
  const numeric=rule.minimum!==undefined&&rule.maximum!==undefined;
  return <div className="field">
    <div className="field-header">
      <label htmlFor={`answer-${name}`}>{info.label}<span className="required-mark"> *</span></label>
      <button
        type="button"
        className="field-info-trigger"
        aria-label={`More information about ${info.label}`}
        title="About this answer"
        onClick={()=>onOpenInfo(name)}
      ><Info size={15}/><span>Info</span></button>
    </div>
    {numeric
      ?<div className="number-entry">
        <input
          id={`answer-${name}`}
          aria-label={info.label}
          type="number"
          min={rule.minimum}
          max={rule.maximum}
          step={rule.type==='integer'?1:0.1}
          value={value??''}
          onChange={event=>onChange(event.target.value===''?undefined:Number(event.target.value))}
          required
        />
          {info.unit&&<span>{info.unit}</span>}
      </div>
      :<select
        id={`answer-${name}`}
        aria-label={info.label}
        value={value===undefined?'':String(value)}
        onChange={event=>onChange(event.target.value===''?undefined:Number(event.target.value))}
        required
      >
        <option value="">Choose the option from your report</option>
        {options.map(option=><option key={option.value} value={option.value}>{option.label}</option>)}
      </select>}
  </div>;
}

function FieldInfoDrawer({
  feature,schema,onClose,
}:{
  feature:FeatureKey|null;
  schema:InputSchema|null;
  onClose:()=>void;
}) {
  const dialogRef=useRef<HTMLDialogElement>(null);
  const info=feature?healthInformation[feature]:null;
  const rule=feature&&schema?schema[feature]:null;
  const numeric=Boolean(rule?.minimum!==undefined&&rule.maximum!==undefined);
  const requiresReport=Boolean(feature&&['cp','restecg','exang','slope','ca','thal','thalach','oldpeak'].includes(feature));

  useEffect(()=>{
    const dialog=dialogRef.current;
    if(!dialog)return;
    if(feature&&!dialog.open)dialog.showModal();
    else if(!feature&&dialog.open)dialog.close();
  },[feature]);

  return <dialog
    ref={dialogRef}
    className="field-info-drawer"
    aria-labelledby="field-info-title"
    onCancel={event=>{event.preventDefault();onClose();}}
    onKeyDown={event=>{if(event.key==='Escape'){event.preventDefault();onClose();}}}
    onClick={event=>{if(event.target===event.currentTarget)onClose();}}
  >
    {feature&&info&&<div className="field-info-content">
      <div className="field-info-heading">
        <div><span className="eyebrow">ABOUT THIS ANSWER</span><h2 id="field-info-title">{info.label}</h2></div>
        <button type="button" className="drawer-close" aria-label="Close information" onClick={onClose}><X size={20}/></button>
      </div>
      <section><h3>What is this?</h3><p>{info.meaning}</p></section>
      <section><h3>Where can I find it?</h3><p>{requiresReport
        ?'Check the matching ECG, exercise-test, or medical report. If you cannot find this value, ask a healthcare professional. Do not guess.'
        :'Use the information in your own records.'}</p></section>
      <section><h3>Why do we ask?</h3><p>CARDIA compares these answers with patterns in older heart-health records. It cannot tell whether a value is normal for you.</p></section>
      {numeric&&rule&&<section className="input-range-note">
        <h3>Values this form accepts</h3>
        <p>{rule.minimum} to {rule.maximum}{info.unit?` ${info.unit}`:''}. These are input limits for the research model, not medical reference ranges.</p>
      </section>}
      <section><h3>Learn more</h3><div className="drawer-sources">{info.sources.map(source=>
        <a key={source.url} href={source.url} target="_blank" rel="noreferrer">{source.label}<ArrowRight size={14}/></a>)}</div></section>
    </div>}
  </dialog>;
}

type HealthSource=typeof healthInformation.age.sources[number];

function SourceDetails({sources,title='Sources and further reading'}:{sources:HealthSource[];title?:string}) {
  const uniqueSources=[...new Map(sources.map(source=>[source.url,source])).values()];
  if(!uniqueSources.length)return null;
  return <details className="source-details">
    <summary>{title}</summary>
    <ul>{uniqueSources.map(source=><li key={source.url}>
      <a href={source.url} target="_blank" rel="noreferrer">{source.label}</a>
      <small>{source.kind==='dataset'?'Study information':'General health information'}</small>
    </li>)}</ul>
  </details>;
}

function InfluenceRow({
  feature,value,contribution,predictedClass,relativeMagnitude,
}:{
  feature:FeatureKey;
  value:number;
  contribution:number;
  predictedClass:0|1;
  relativeMagnitude:number;
}) {
  const info=healthInformation[feature];
  const movedToward=predictedClass===1?contribution>0:contribution<0;
  const direction=contribution===0?'No measurable shift'
    :movedToward?'Toward this result'
    :'Away from this result';
  return <details className="influence-row">
    <summary>
      <span className="influence-main">
        <span className="influence-title">{info.label}</span>
        <strong>{formatFeatureValue(feature,value)}</strong>
      </span>
      <span className="influence-direction">{direction}</span>
      <span
        className="influence-track"
        role="img"
        aria-label={`Relative influence: ${relativeMagnitude.toFixed(0)} percent; ${direction.toLowerCase()}`}
      ><i className={contribution===0?'neutral':movedToward?'toward':'away'} style={{width:`${relativeMagnitude}%`}}/></span>
      <ChevronDown className="influence-chevron" size={16} aria-hidden="true"/>
    </summary>
    <div className="influence-detail">
      <p>{info.meaning}</p>
      <p><b>Why it may matter:</b> {info.relevance}</p>
      <p><b>What this cannot tell you:</b> {info.limitations}</p>
      <p className="influence-safety">This describes how the AI used your answer. It is not a medical cause or diagnosis.</p>
    </div>
  </details>;
}

function AssessmentResult({
  record,onViewHistory,
}:{
  record:AssessmentRecord;
  onViewHistory?:()=>void;
}) {
  const ranked=record.explanation&&record.inputs
    ?[...record.explanation.contributions].sort((a,b)=>Math.abs(b.value)-Math.abs(a.value))
    :[];
  const maxContribution=Math.max(...ranked.map(item=>Math.abs(item.value)),1e-12);
  const selectedProbability=record.predictedClass===1?record.probabilities.class1:record.probabilities.class0;
  const otherProbability=record.predictedClass===1?record.probabilities.class0:record.probabilities.class1;
  const selectedPattern=record.predictedClass===1
    ?'Pattern linked to heart-disease records in the study'
    :'Other pattern in the study records';
  const resultSources=featureKeys.flatMap(feature=>healthInformation[feature].sources);
  return <section className="panel result-panel" aria-live="polite">
    <div className="result-heading">
      <div>
        <span className="eyebrow">YOUR RESULT</span>
        <h2>Your assessment result</h2>
        <p>{patternLabel(record)}</p>
      </div>
      <div className="result-mark"><Check size={22}/></div>
    </div>
    <div className="result-score">
      <div>
        <span>AI estimate for this study pattern</span>
        <strong>{(selectedProbability*100).toFixed(1)}%</strong>
        <small>{selectedPattern}</small>
      </div>
      <p>This is an educational AI estimate, not a medical diagnosis or your personal chance of having heart disease.</p>
    </div>
    <section className="result-section">
      <h3>How did the AI reach this result?</h3>
      <p>The AI compared your answers with patterns in older heart-health records. It used those patterns to prepare this result; it cannot confirm a health condition.</p>
      <ol className="result-process">
        <li><b>You enter answers</b><span>Use information from your records.</span></li>
        <li><b>The AI compares patterns</b><span>It looks for patterns found in the study records.</span></li>
        <li><b>CARDIA explains the result</b><span>You can review which answers influenced the computer.</span></li>
      </ol>
    </section>
    <section className="result-section">
      <h3>What influenced your result?</h3>
      {record.explanation&&record.inputs
        ?<>
          <p>The AI looked at your answers and measurements. These had the strongest influence on its result. The bars compare their relative influence; they are not health-risk scores.</p>
          <div className="influence-list">
            {ranked.slice(0,6).map(item=><InfluenceRow
              key={item.feature}
              feature={item.feature}
              value={record.inputs![item.feature]}
              contribution={item.value}
              predictedClass={record.predictedClass}
              relativeMagnitude={Math.abs(item.value)/maxContribution*100}
            />)}
          </div>
          <details className="all-answers">
            <summary>See all 13 answers</summary>
            <dl className="answers-grid">{featureKeys.map(feature=><div key={feature}>
              <dt>{labels[feature]}</dt>
              <dd>{formatFeatureValue(feature,record.inputs![feature])}</dd>
            </div>)}</dl>
          </details>
        </>
        :<p className="explanation-error">We could not show which answers influenced this result. The estimate is still shown above, but it is for learning only.</p>}
    </section>
    <section className="result-section next-steps">
      <h3>What to do next</h3>
      <ul>
        <li>Discuss this research result with a qualified healthcare professional.</li>
        <li>Bring the original blood pressure, blood test, ECG, exercise-test, or imaging reports used for these answers.</li>
        <li>Do not start, stop, or change medication based on CARDIA.</li>
      </ul>
      <p className="emergency-warning">If you have severe or new chest pain, difficulty breathing, fainting, or other possible emergency symptoms, seek immediate medical attention. Do not wait for this application.</p>
    </section>
    <SourceDetails sources={resultSources}/>
    <details className="technical-details">
      <summary>Technical details (optional)</summary>
      <p>This section is for students and researchers. The values describe how the AI calculated its result; they do not explain a medical cause.</p>
      <dl className="technical-summary">
        <div><dt>Model</dt><dd>{record.modelName}</dd></div>
        <div><dt>Model version</dt><dd>{record.modelVersion}</dd></div>
        <div><dt>Decision cutoff</dt><dd>{(record.decisionThreshold*100).toFixed(0)}%</dd></div>
        <div><dt>Selected study-pattern score</dt><dd>{(selectedProbability*100).toFixed(1)}%</dd></div>
        <div><dt>Other study-pattern score</dt><dd>{(otherProbability*100).toFixed(1)}%</dd></div>
        <div><dt>Study information</dt><dd>Small, historical Cleveland heart-health records; not a clinical validation.</dd></div>
      </dl>
      {record.explanation&&record.inputs&&<>
        <h4>Local feature contributions</h4>
        <p>These SHAP contributions are shown in log-odds. Positive values move the model toward its heart-disease study category; negative values move it away. They are model calculations, not medical causes.</p>
        <p><b>Reference output:</b> {record.explanation.base_value.toFixed(4)}</p>
        <div className="technical-contributions-wrap">
          <table className="technical-contributions">
            <thead><tr><th>Original input name</th><th>Entered answer</th><th>Submitted code/value</th><th>SHAP contribution</th></tr></thead>
            <tbody>{ranked.map(item=><tr key={item.feature}>
              <td>{item.feature}</td>
              <td>{formatFeatureValue(item.feature,record.inputs![item.feature])}</td>
              <td>{record.inputs![item.feature]}</td>
              <td>{item.value.toFixed(5)}</td>
            </tr>)}</tbody>
          </table>
        </div>
        <small>{record.explanation.note}</small>
      </>}
      {record.explanationError&&<p>Explanation status: {record.explanationError}</p>}
    </details>
    {onViewHistory&&<button className="secondary-button result-history-link" onClick={onViewHistory}>
      <History size={16}/> View or download this saved assessment
    </button>}
  </section>;
}

function App() {
  const [demo]=useState(loadDemoState);
  const [userId]=useState(demo.userId);
  const [assessments,setAssessments]=useState(demo.assessments);
  const [profileName,setProfileName]=useState(demo.profileName);
  const [storageWarning,setStorageWarning]=useState(demo.warning);
  const [page,setPage]=useState<Page>('home');
  const [patient,setPatient]=useState<PatientForm>({});
  const [step,setStep]=useState(0);
  const [resultRecord,setResultRecord]=useState<AssessmentRecord|null>(null);
  const [loading,setLoading]=useState(false);
  const [error,setError]=useState('');
  const [formErrors,setFormErrors]=useState<string[]>([]);
  const [serviceError,setServiceError]=useState('');
  const [model,setModel]=useState<ModelInfo|null>(null);
  const [inputSchema,setInputSchema]=useState<InputSchema|null>(null);
  const [health,setHealth]=useState<'checking'|'online'|'offline'>('checking');
  const [mobileNav,setMobileNav]=useState(false);
  const [chartFeature,setChartFeature]=useState<NumericFeature>('trestbps');
  const [selectedRecordId,setSelectedRecordId]=useState<string|null>(demo.assessments[0]?.id??null);
  const [historyMessage,setHistoryMessage]=useState('');

  const checkService=async()=>{
    setHealth('checking');
    setServiceError('');
    try{
      const [healthInfo,modelInfo,schema]=await Promise.all([
        apiRequest('/api/health'),
        apiRequest('/api/model'),
        apiRequest('/api/input-schema'),
      ]);
      if(typeof healthInfo!=='object'||healthInfo===null||!('status'in healthInfo)||(healthInfo as Record<string,unknown>).status!=='ok'){
        throw new Error('The CARDIA service is not ready yet.');
      }
      if(!isModelInfo(modelInfo))throw new Error('CARDIA could not load its study information.');
      if(!isInputSchema(schema))throw new Error('CARDIA could not load answer guidance for all questions.');
      setModel(modelInfo);
      setInputSchema(schema);
      setHealth('online');
    }catch(cause){
      setModel(null);
      setInputSchema(null);
      setHealth('offline');
      setServiceError(cause instanceof Error?cause.message:'CARDIA could not connect to its service.');
    }
  };

  useEffect(()=>{void checkService();},[]);

  const latestRecord=assessments[0]??null;
  const selectedRecord=assessments.find(record=>record.id===selectedRecordId)??latestRecord;
  const recentRecords=assessments.slice(0,4);
  const progress=(step+1)/steps.length*100;
  const currentKeys=steps[step].keys;

  const updatePatient=(key:FeatureKey,value:number|undefined)=>{
    setPatient(previous=>{
      const next={...previous};
      if(value===undefined)delete next[key];
      else next[key]=value;
      return next;
    });
    setFormErrors([]);
    setError('');
    setResultRecord(null);
  };

  const storeAssessment=(record:AssessmentRecord)=>{
    const next=[record,...assessments].slice(0,100);
    setAssessments(next);
    setSelectedRecordId(record.id);
    setResultRecord(record);
    try{
      saveDemoAssessments(next);
      setStorageWarning('');
    }catch(cause){
      setStorageWarning(`Your result is available, but this browser could not save it: ${cause instanceof Error?cause.message:'storage access failed'}`);
    }
  };

  const runAssessment=async()=>{
    if(!inputSchema){
      setError('Answer guidance is unavailable. Reconnect to CARDIA before submitting.');
      return;
    }
    const issues=validatePatient(patient,inputSchema);
    setFormErrors(issues);
    if(issues.length)return;
    if(!isPatient(patient)){
      setError('Please complete every answer before submitting.');
      return;
    }
    setLoading(true);
    setError('');
    try{
      const payload:Patient=patient;
      const response=await apiRequest('/api/predict',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(payload),
      });
      if(!isPrediction(response))throw new Error('The prediction service returned an incomplete result.');
      const class1=response.probability;
      let explanation:Explanation|null=null;
      let explanationError='';
      try{
        const explanationResponse=await apiRequest('/api/explain',{
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify(payload),
        });
        if(!isExplanation(explanationResponse)){
          throw new Error('The explanation service returned incomplete feature information.');
        }
        explanation=explanationResponse;
      }catch(cause){
        explanationError=cause instanceof Error?cause.message:'The explanation service failed.';
      }
      const record:AssessmentRecord={
        schemaVersion:2,
        id:crypto.randomUUID(),
        userId,
        createdAt:new Date().toISOString(),
        inputs:{...payload},
        modelName:response.model_name,
        modelVersion:response.model_version,
        predictedClass:class1>=classificationThreshold?1:0,
        probabilities:{class0:1-class1,class1},
        decisionThreshold:classificationThreshold,
        explanation,
        ...(explanationError?{explanationError}:{}),
      };
      storeAssessment(record);
      setPage('assessment');
      setStep(3);
    }catch(cause){
      setError(cause instanceof Error?cause.message:'Unable to complete this assessment.');
    }finally{
      setLoading(false);
    }
  };

  const updateProfileName=(name:string)=>{
    setProfileName(name);
    try{
      saveDemoProfileName(name);
      setStorageWarning('');
    }catch(cause){
      setStorageWarning(`The name is shown for this session, but could not be saved in this browser: ${cause instanceof Error?cause.message:'storage access failed'}`);
    }
  };

  const clearHistory=()=>{
    setAssessments([]);
    setSelectedRecordId(null);
    setHistoryMessage('');
    try{
      clearDemoAssessments();
      setStorageWarning('');
    }catch(cause){
      setStorageWarning(`Could not remove saved browser data: ${cause instanceof Error?cause.message:'storage access failed'}`);
    }
  };

  const deleteRecord=(id:string)=>{
    const next=assessments.filter(record=>record.id!==id);
    setAssessments(next);
    setSelectedRecordId(next[0]?.id??null);
    try{
      saveDemoAssessments(next);
      setHistoryMessage('Assessment deleted from this browser.');
      setStorageWarning('');
    }catch(cause){
      setHistoryMessage(`Could not save this deletion: ${cause instanceof Error?cause.message:'storage access failed'}`);
    }
  };

  const exportSummary=(record:AssessmentRecord)=>{
    try{
      const lines=[
        'CARDIA assessment summary',
        `Date: ${dateLabel(record.createdAt,true)}`,
        '',
        'AI estimate (for learning only; not a diagnosis or your personal chance of heart disease):',
        `Pattern linked to heart-disease records in the study: ${(record.probabilities.class1*100).toFixed(1)}%`,
        `Other pattern in the study records: ${(record.probabilities.class0*100).toFixed(1)}%`,
        '',
        'Submitted information:',
        ...(record.inputs?featureKeys.map(key=>`- ${labels[key]}: ${formatFeatureValue(key,record.inputs![key])}`):['- Detailed answers were not saved with this older assessment.']),
        '',
        'What influenced the AI result:',
        ...(record.explanation
          ?[...record.explanation.contributions]
            .sort((a,b)=>Math.abs(b.value)-Math.abs(a.value))
            .slice(0,6)
            .map(item=>{
              const toward=record.predictedClass===1?item.value>0:item.value<0;
              return `- ${labels[item.feature]} moved the model ${toward?'toward':'away from'} its selected result. This is not a medical cause.`;
            })
          :['- No verified explanation was saved for this assessment.']),
        '',
        'Use the original health reports and discuss them with a qualified healthcare professional. Do not start, stop, or change medication based on CARDIA.',
      ];
      const blob=new Blob([lines.join('\n')],{type:'text/plain;charset=utf-8'});
      const url=URL.createObjectURL(blob);
      const anchor=document.createElement('a');
      anchor.href=url;
      anchor.download=`cardia-assessment-${record.id}.txt`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(()=>URL.revokeObjectURL(url),1000);
      setHistoryMessage('Assessment summary downloaded.');
    }catch(cause){
      setHistoryMessage(`Could not export this summary: ${cause instanceof Error?cause.message:'download failed'}`);
    }
  };

  const openHistory=(id:string)=>{
    setSelectedRecordId(id);
    setPage('history');
  };

  const startAssessment=()=>{
    setPatient({});
    setFormErrors([]);
    setError('');
    setResultRecord(null);
    setStep(0);
    setPage('assessment');
  };

  const continueStep=()=>{
    if(!inputSchema)return;
    const issues=validatePatient(patient,inputSchema,currentKeys);
    setFormErrors(issues);
    if(!issues.length)setStep(current=>Math.min(current+1,steps.length-1));
  };

  const cvRows=useMemo(()=>model?.cv_table??[],[model]);

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav?'open':''}`}>
      <div className="brand">
        <div className="brand-mark"><HeartPulse size={22}/></div>
        <div><b>cardia</b><small>HEART HEALTH</small></div>
      </div>
      <div className="workspace-label">YOUR SPACE</div>
      <nav aria-label="Main navigation">
        {navItems.map(item=><button
          key={item.id}
          className={`nav-item ${page===item.id?'active':''}`}
          aria-current={page===item.id?'page':undefined}
          onClick={()=>{setPage(item.id);setMobileNav(false);}}
        >
          <item.icon size={18}/><span>{item.label}</span>
        </button>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="privacy-note"><ShieldCheck size={17}/><div>
          <b>Browser demo only</b><small>Use fictional information, not real patient data.</small>
        </div></div>
        <div className="user-chip"><div className="avatar">{profileName.trim()?profileName.trim().slice(0,2).toUpperCase():'CA'}</div>
          <div><b>{profileName.trim()||'CARDIA user'}</b><small>Saved in this browser only</small></div>
          <ChevronDown size={15}/>
        </div>
      </div>
    </aside>
    <main className="main-area">
      <header className="topbar">
        <button className="mobile-menu" aria-label="Open navigation" onClick={()=>setMobileNav(!mobileNav)}><Menu/></button>
        <div className="breadcrumb">CARDIA <span>/</span> <b>{navItems.find(item=>item.id===page)?.label}</b></div>
        <div className="top-actions">
          <span className={`status ${health}`}><i/>{health==='online'?'Connected':health==='offline'?"Can't connect":'Connecting'}</span>
          {health==='offline'&&<button className="text-button" onClick={()=>void checkService()}>Retry</button>}
        </div>
      </header>
      <div className="content">
        {storageWarning&&<div className="storage-warning" role="status">{storageWarning}</div>}
        {health==='offline'&&<div className="service-inline-error" role="status">
          {serviceError||"CARDIA couldn't connect. Check the service and try again."}
        </div>}
        <AnimatePresence mode="wait">
          <motion.div key={page} initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-5}} transition={{duration:.18}}>
            {page==='home'&&<HomePage
              profileName={profileName}
              assessments={assessments}
              chartFeature={chartFeature}
              onChartFeatureChange={setChartFeature}
              onProfileNameChange={updateProfileName}
              onStart={startAssessment}
              onLearn={()=>setPage('learning')}
              onOpenRecord={openHistory}
            />}
            {page==='assessment'&&<AssessmentPage
              patient={patient}
              step={step}
              schema={inputSchema}
              error={error}
              formErrors={formErrors}
              health={health}
              loading={loading}
              result={resultRecord}
              onChange={updatePatient}
              onStepChange={setStep}
              onContinue={continueStep}
              onSubmit={()=>void runAssessment()}
              onReset={startAssessment}
              onOpenSaved={()=>resultRecord&&openHistory(resultRecord.id)}
            />}
            {page==='history'&&<HistoryPage
              records={assessments}
              selected={selectedRecord}
              message={historyMessage}
              onSelect={setSelectedRecordId}
              onDelete={deleteRecord}
              onClear={clearHistory}
              onExport={exportSummary}
              onStart={startAssessment}
            />}
            {page==='learning'&&<LearningPage/>}
            {page==='about'&&<AboutPage model={model} cvRows={cvRows}/>}
          </motion.div>
        </AnimatePresence>
        <footer>CARDIA <span>Educational heart-health research demo</span></footer>
      </div>
    </main>
  </div>;
}

function HomePage({
  profileName,assessments,chartFeature,onChartFeatureChange,onProfileNameChange,onStart,onLearn,onOpenRecord,
}:{
  profileName:string;
  assessments:AssessmentRecord[];
  chartFeature:NumericFeature;
  onChartFeatureChange:(feature:NumericFeature)=>void;
  onProfileNameChange:(name:string)=>void;
  onStart:()=>void;
  onLearn:()=>void;
  onOpenRecord:(id:string)=>void;
}) {
  const latest=assessments[0];
  return <>
    <section className="home-welcome">
      <div>
        <span className="eyebrow">YOUR HEART HEALTH SPACE</span>
        <h1>{profileName.trim()?`Welcome, ${profileName.trim()}.`:'Welcome to your heart health space.'}</h1>
        <p>Understand your health information and explore what this research model can tell you.</p>
      </div>
      <button className="primary-button" onClick={onStart}><ClipboardList size={17}/> Start a heart health check</button>
    </section>
    <div className="profile-name-row">
      <label htmlFor="profile-name">Name for this browser demo</label>
      <input id="profile-name" value={profileName} maxLength={32} placeholder="Optional" onChange={event=>onProfileNameChange(event.target.value)}/>
    </div>
    <div className="home-summary">
      <div><span>COMPLETED CHECKS</span><strong>{assessments.length}</strong><small>Saved in this browser</small></div>
      <div><span>LATEST CHECK</span><strong>{latest?dateLabel(latest.createdAt):'Not yet'}</strong><small>{latest?patternLabel(latest):'Your first check will appear here'}</small></div>
      <div><span>YOUR NEXT STEP</span><strong>{latest?'Review your history':'Start here'}</strong><small>Use only fictional information in this demo</small></div>
    </div>
    <div className="notice">
      <div className="notice-icon"><ShieldCheck size={19}/></div>
      <div><b>For learning only</b><p>This AI result is not a medical diagnosis or a replacement for professional care. Use fictional information in this demo.</p></div>
    </div>
    <div className="home-activity-grid">
      <Suspense fallback={<div className="panel history-empty chart-empty" role="status">Loading measurement history…</div>}>
        <HistoryChart records={assessments} feature={chartFeature} onFeatureChange={onChartFeatureChange}/>
      </Suspense>
      <section className="panel recent-panel">
        <div className="panel-head"><div><h2>Recent activity</h2><p>Your saved checks, newest first</p></div><History size={19}/></div>
        {assessments.length
          ?<div className="recent-list">{assessments.slice(0,4).map(record=><button className="recent-row" key={record.id} onClick={()=>onOpenRecord(record.id)}>
            <CalendarDays size={17}/>
            <span><b>{dateLabel(record.createdAt)}</b><small>{record.inputs?patternLabel(record):'Older summary; answers were not stored'}</small></span>
            <ArrowRight size={15}/>
          </button>)}</div>
          :<div className="history-empty">
            <HeartPulse size={26}/><h3>No checks yet</h3>
            <p>You have not completed a check yet. Start your first demo check to see its date and answers here.</p>
            <button className="text-button" onClick={onStart}>Start your first check <ArrowRight size={14}/></button>
          </div>}
      </section>
    </div>
    <section className="learning-strip">
      <div className="edu-icon"><BookOpen size={20}/></div>
      <div><h2>Understand your health information</h2><p>Learn about blood pressure, cholesterol, heart tests, and how this research model works.</p></div>
      <button className="text-button" onClick={onLearn}>Open the heart health guide <ArrowRight size={14}/></button>
    </section>
  </>;
}

function AssessmentPage({
  patient,step,schema,error,formErrors,health,loading,result,onChange,onStepChange,onContinue,onSubmit,onReset,onOpenSaved,
}:{
  patient:PatientForm;
  step:number;
  schema:InputSchema|null;
  error:string;
  formErrors:string[];
  health:'checking'|'online'|'offline';
  loading:boolean;
  result:AssessmentRecord|null;
  onChange:(key:FeatureKey,value:number|undefined)=>void;
  onStepChange:(step:number)=>void;
  onContinue:()=>void;
  onSubmit:()=>void;
  onReset:()=>void;
  onOpenSaved:()=>void;
}) {
  const [infoFeature,setInfoFeature]=useState<FeatureKey|null>(null);
  const stepIndex=(feature:FeatureKey)=>steps.findIndex(item=>item.keys.includes(feature));
  return <>
    <div className="page-heading">
      <div><span className="eyebrow">GUIDED CHECK · FICTIONAL DATA ONLY</span><h1>Heart health check</h1><p>Enter information as it appears in your records. Do not guess missing test results.</p></div>
      <button className="secondary-button" onClick={onReset}><RotateCcw size={16}/> Start over</button>
    </div>
    <div className="assessment-layout assessment-single-column">
      <section className="panel form-panel">
        <div className="step-top">
          <div><span className="step-count">STEP {step+1} OF {steps.length}</span><h2>{steps[step].title}</h2><p>{steps[step].description}</p></div>
          <div className="step-symbol"><ClipboardList size={22}/></div>
        </div>
        <div className="progress-track"><motion.div className="progress-fill" animate={{width:`${(step+1)/steps.length*100}%`}}/></div>
        {step===2&&<div className="assessment-guidance"><ShieldCheck size={17}/><span>Use the exact entries from your report. If you are unsure, ask a healthcare professional instead of guessing.</span></div>}
        {step<3
          ?<div className="field-grid">{schema
            ?steps[step].keys.map(key=><Field key={key} name={key} value={patient[key]} schema={schema} onChange={value=>onChange(key,value)} onOpenInfo={setInfoFeature}/>)
            :<div className="service-inline-error">Answer guidance is unavailable. Reconnect to CARDIA before entering answers.</div>}</div>
          :<div className="review-grid">{featureKeys.map(key=><div className="review-item" key={key}>
            <span>{labels[key]}</span>
            <b>{patient[key]===undefined?'Not entered':formatFeatureValue(key,patient[key])}</b>
            <button type="button" className="text-button edit-answer" onClick={()=>onStepChange(Math.max(0,stepIndex(key)))}>Edit</button>
          </div>)}</div>}
        {formErrors.length>0&&<div className="error-box" role="alert">
          <b>Please check these answers:</b><ul>{formErrors.map(issue=><li key={issue}>{issue}</li>)}</ul>
        </div>}
        {error&&<div className="error-box" role="alert">{error}</div>}
        {step===3&&health!=='online'&&<div className="error-box" role="status">Reconnect to CARDIA before submitting. Use Retry at the top of the page.</div>}
        <div className="form-footer">
          <button className="secondary-button" disabled={step===0||loading} onClick={()=>onStepChange(step-1)}><ArrowLeft size={16}/> Back</button>
          {step<3
            ?<button className="primary-button" disabled={!schema||loading} onClick={onContinue}>Continue <ArrowRight size={16}/></button>
            :<button className="primary-button" disabled={loading||health!=='online'||!schema} onClick={onSubmit}>
              {loading?<LoaderCircle className="spin" size={17}/>:<Activity size={17}/>}
              {loading?'Preparing your result…':'Show my result'}
            </button>}
        </div>
      </section>
    </div>
    <FieldInfoDrawer feature={infoFeature} schema={schema} onClose={()=>setInfoFeature(null)}/>
    {result&&<AssessmentResult record={result} onViewHistory={onOpenSaved}/>}
  </>;
}

function HistoryPage({
  records,selected,message,onSelect,onDelete,onClear,onExport,onStart,
}:{
  records:AssessmentRecord[];
  selected:AssessmentRecord|null;
  message:string;
  onSelect:(id:string)=>void;
  onDelete:(id:string)=>void;
  onClear:()=>void;
  onExport:(record:AssessmentRecord)=>void;
  onStart:()=>void;
}) {
  return <>
    <div className="page-heading">
      <div><span className="eyebrow">BROWSER DEMO RECORDS</span><h1>Assessment history</h1><p>Revisit saved demonstration results and download a plain-language summary.</p></div>
      {records.length>0&&<button className="secondary-button" onClick={onClear}><Trash2 size={15}/> Delete all demo records</button>}
    </div>
    <div className="notice"><div className="notice-icon"><ShieldCheck size={19}/></div><div>
      <b>Stored in this browser only</b><p>This is not a secure medical record or a signed-in account. Do not save real patient information.</p>
    </div></div>
    {message&&<div className="storage-warning" role="status">{message}</div>}
    {records.length
      ?<div className="history-page-grid">
        <section className="panel history-panel">
          <h2>Saved checks</h2>
          <div className="history-list">{records.map(record=><button
            className={`history-row history-select ${selected?.id===record.id?'selected':''}`}
            key={record.id}
            onClick={()=>onSelect(record.id)}
          >
            <div className="history-icon"><HeartPulse size={18}/></div>
            <div className="history-main"><b>{dateLabel(record.createdAt,true)}</b><small>{patternLabel(record)}</small></div>
            <ArrowRight size={15}/>
          </button>)}</div>
        </section>
        <section className="history-record-detail">
          {selected
            ?<><div className="history-detail-actions">
              <button className="secondary-button" onClick={()=>onExport(selected)}><Download size={15}/> Download summary</button>
              <button className="text-button delete-record" onClick={()=>onDelete(selected.id)}><Trash2 size={15}/> Delete this check</button>
            </div>
            {selected.inputs
              ?<AssessmentResult record={selected}/>
              :<section className="panel result-panel">
                <h2>Older saved assessment</h2><p>{dateLabel(selected.createdAt,true)}</p>
                <p>{patternLabel(selected)}</p>
                <p>This older entry saved only your age and result. The other answers and explanation were not stored, so they cannot be shown.</p>
                <div className="class-output-grid">
                  <div><span>Pattern linked to heart-disease cases in the original study</span><strong>{(selected.probabilities.class1*100).toFixed(1)}%</strong></div>
                  <div><span>Other pattern in the original study</span><strong>{(selected.probabilities.class0*100).toFixed(1)}%</strong></div>
                </div>
              </section>}</>
            :<div className="panel history-empty"><p>Select a saved check to view it.</p></div>}
        </section>
      </div>
      :<div className="panel history-empty history-empty-large">
        <History size={30}/><h2>No saved checks yet</h2><p>You have not completed an assessment yet. Start your first check to see your result and answers here.</p>
        <button className="primary-button" onClick={onStart}>Start a heart health check <ArrowRight size={15}/></button>
      </div>}
  </>;
}

function LearningPage() {
  const topics:Array<{title:string;feature:FeatureKey;description?:string;source?:typeof AI_HEALTH_SOURCE}>=[
    {title:'Understanding blood pressure',feature:'trestbps'},
    {title:'Understanding cholesterol',feature:'chol'},
    {title:'Understanding blood sugar',feature:'fbs'},
    {title:'Understanding heartbeat',feature:'thalach'},
    {title:'What is an ECG?',feature:'restecg'},
    {title:'What an exercise heart test records',feature:'thalach'},
    {title:'Understanding chest discomfort',feature:'cp'},
    {
      title:'How this computer model works',feature:'age',source:AI_HEALTH_SOURCE,
      description:'The computer compares the information entered with patterns in older heart-health records. It does not know a person’s full health history.',
    },
    {
      title:'What a computer prediction can tell you',feature:'age',source:AI_HEALTH_SOURCE,
      description:'The result shows which pattern the computer matched to the answers entered. It cannot confirm a health condition or predict what will happen to one person.',
    },
    {
      title:'Why an AI result is not a diagnosis',feature:'age',source:AI_HEALTH_SOURCE,
      description:'A healthcare professional considers symptoms, medical history, and test results together. CARDIA cannot confirm a condition or replace that assessment.',
    },
  ];
  return <>
    <div className="page-heading"><div><span className="eyebrow">HEART HEALTH GUIDE</span><h1>Learn at your own pace.</h1><p>Short explanations of the information used in this research demo.</p></div></div>
    <div className="education-grid">{topics.map(topic=>{
      const info=healthInformation[topic.feature];
      return <article className="panel education-card" key={topic.title}>
        <div className="edu-icon"><BookOpen size={20}/></div>
        <h2>{topic.title}</h2>
        <p>{topic.description??info.meaning}</p>
        {!topic.description&&<p>{info.limitations}</p>}
      </article>;
    })}</div>
    <SourceDetails sources={topics.flatMap(topic=>
      topic.source?[topic.source]:healthInformation[topic.feature].sources
    )}/>
    <div className="notice"><div className="notice-icon"><Stethoscope size={19}/></div><div><b>Get personal medical questions answered by a healthcare professional</b><p>This guide is general information, not a diagnosis or a treatment plan.</p></div></div>
  </>;
}

function AboutPage({model,cvRows}:{model:ModelInfo|null;cvRows:Array<Record<string,unknown>>}) {
  const metrics=model?.test_metrics??{};
  const metric=(key:string)=>typeof metrics[key]==='number'?Number(metrics[key]):null;
  const count=(key:string)=>metric(key)?.toFixed(0)??'—';
  return <>
    <div className="page-heading"><div><span className="eyebrow">ABOUT THIS DEMO</span><h1>About CARDIA</h1><p>A learning project that explores how computer models process health information.</p></div></div>
    <section className="panel about-panel">
      <h2>What CARDIA is for</h2>
      <p>CARDIA is a learning project that compares entered answers with a small collection of older heart-health records. It has not been tested to tell someone whether they have heart disease or what care they need.</p>
      <h3>Important limits</h3>
      <ul className="limitations">
        <li>The records are limited and may not represent people today.</li>
        <li>The computer's percentage is not your personal chance of having heart disease.</li>
        <li>An explanation shows how the computer used your answers. It does not show what caused a health condition.</li>
        <li>Use fictional information only. This demo is saved in this browser and is not secure medical-record storage.</li>
      </ul>
      <details className="advanced-details">
        <summary>Technical details (optional)</summary>
        <p>This section is for students and developers. CARDIA uses {model?.selected_model||'the configured model'} and a {classificationThreshold.toFixed(2)} decision cutoff. Its local SHAP calculations are checked against the model's output.</p>
        <div className="advanced-metrics">
          <div><span>ROC–AUC</span><b>{metric('roc_auc')?.toFixed(3)??'—'}</b></div>
          <div><span>Accuracy</span><b>{metric('accuracy')!==null?`${(metric('accuracy')!*100).toFixed(1)}%`:'—'}</b></div>
          <div><span>Sensitivity</span><b>{metric('recall_sensitivity')!==null?`${(metric('recall_sensitivity')!*100).toFixed(1)}%`:'—'}</b></div>
          <div><span>Specificity</span><b>{metric('specificity')!==null?`${(metric('specificity')!*100).toFixed(1)}%`:'—'}</b></div>
          <div><span>Precision</span><b>{metric('precision')!==null?`${(metric('precision')!*100).toFixed(1)}%`:'—'}</b></div>
          <div><span>Records in project split</span><b>{model?.n_test??'—'}</b></div>
          <div><span>Correct negative labels (TN)</span><b>{count('tn')}</b></div>
          <div><span>Incorrect positive labels (FP)</span><b>{count('fp')}</b></div>
          <div><span>Missed positive labels (FN)</span><b>{count('fn')}</b></div>
          <div><span>Correct positive labels (TP)</span><b>{count('tp')}</b></div>
        </div>
        <p>The current project evaluation contains only {model?.n_test??'—'} records. Earlier candidate comparisons used this split, so these figures are exploratory and are not an independent clinical validation.</p>
        <h3>Candidate model comparison</h3>
        <div className="technical-contributions-wrap">
          <table className="technical-contributions">
            <thead><tr><th>Model</th><th>Mean cross-validation ROC–AUC</th><th>Standard deviation</th></tr></thead>
            <tbody>{cvRows.map((row,index)=><tr key={`${String(row.model)}-${index}`}>
              <td>{String(row.model??'')}</td>
              <td>{typeof row.roc_auc_mean==='number'?row.roc_auc_mean.toFixed(3):'—'}</td>
              <td>{typeof row.roc_auc_std==='number'?row.roc_auc_std.toFixed(3):'—'}</td>
            </tr>)}</tbody>
          </table>
        </div>
      </details>
    </section>
  </>;
}

export default App;
