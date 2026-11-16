import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { AssessmentRecord } from './assessment';
import { healthInformation } from './healthInformation';

export type NumericFeature='trestbps'|'chol'|'thalach'|'oldpeak';

const measurements:{key:NumericFeature;label:string;unit:string}[]=[
  {key:'trestbps',label:healthInformation.trestbps.label,unit:healthInformation.trestbps.unit||''},
  {key:'chol',label:healthInformation.chol.label,unit:healthInformation.chol.unit||''},
  {key:'thalach',label:healthInformation.thalach.label,unit:healthInformation.thalach.unit||''},
  {key:'oldpeak',label:healthInformation.oldpeak.label,unit:''},
];

function dateLabel(value:string):string {
  const date=new Date(value);
  return Number.isFinite(date.getTime())
    ?date.toLocaleDateString(undefined,{dateStyle:'medium'})
    :'Date unavailable';
}

export default function HistoryChart({
  records,feature,onFeatureChange,
}:{
  records:AssessmentRecord[];
  feature:NumericFeature;
  onFeatureChange:(feature:NumericFeature)=>void;
}) {
  const selected=measurements.find(item=>item.key===feature)!;
  const chartData=records.filter(record=>record.inputs)
    .slice().reverse()
    .map(record=>({
      date:dateLabel(record.createdAt),
      value:record.inputs?.[feature]??null,
    }));

  return <section className="panel chart-panel">
    <div className="panel-head">
      <div><h2>Measurement history</h2><p>Your submitted readings over time. The chart does not judge them as normal or abnormal.</p></div>
      <label className="chart-select-label">Choose a measurement
        <select value={feature} onChange={event=>{
          const next=measurements.find(item=>item.key===event.target.value);
          if(next)onFeatureChange(next.key);
        }}>
          {measurements.map(item=><option key={item.key} value={item.key}>{item.label}</option>)}
        </select>
      </label>
    </div>
    {chartData.length
      ?<div className="history-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{top:15,right:12,bottom:3,left:0}}>
            <CartesianGrid stroke="#e8efeb" strokeDasharray="3 3"/>
            <XAxis dataKey="date" tick={{fill:'#71817E',fontSize:11}}/>
            <YAxis unit={selected.unit?` ${selected.unit}`:''} domain={['auto','auto']} tick={{fill:'#71817E',fontSize:11}} width={62}/>
            <Tooltip formatter={(value)=>[`${value}${selected.unit?` ${selected.unit}`:''}`,selected.label]} labelFormatter={label=>String(label)}/>
            <Line type="monotone" dataKey="value" name={selected.label} stroke="#347F73" strokeWidth={3} dot={{r:4,fill:'#347F73'}} activeDot={{r:6}} connectNulls={false}/>
          </LineChart>
        </ResponsiveContainer>
      </div>
      :<div className="history-empty chart-empty">
        <p>After you save assessments with this measurement, its values will appear here.</p>
      </div>}
  </section>;
}
