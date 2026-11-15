'use strict';
// BEST Lab v1.0 — Metropolis-within-Gibbs (random-walk proposals, transformed scales)
const logGamma=z=>{const p=[0.99999999999980993,676.5203681218851,-1259.1392167224028,771.3234287776531,-176.6150291621406,12.507343278686905,-0.13857109526572012,9.984369578019572e-6,1.5056327351493116e-7];if(z<.5)return Math.log(Math.PI)-Math.log(Math.sin(Math.PI*z))-logGamma(1-z);z-=1;let x=p[0];for(let i=1;i<p.length;i++)x+=p[i]/(z+i);let t=z+7.5;return .5*Math.log(2*Math.PI)+(z+.5)*Math.log(t)-t+Math.log(x)};
function rng(seed){let s=seed>>>0;return ()=>{s+=0x6D2B79F5;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296}}
function normal(r){return Math.sqrt(-2*Math.log(Math.max(1e-12,r())))*Math.cos(2*Math.PI*r())}
function sd(a){const m=a.reduce((s,v)=>s+v,0)/a.length;return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/(a.length-1))}
function posterior(q,a,b,mean,scale){let [m1,m2,l1,l2,ln]=q;let s1=Math.exp(l1),s2=Math.exp(l2),nu=1+Math.exp(ln);if(!Number.isFinite(nu)||nu>1e7||s1<scale/1000||s1>scale*1000||s2<scale/1000||s2>scale*1000)return -Infinity;
 const lpT=(x,m,s)=>logGamma((nu+1)/2)-logGamma(nu/2)-.5*Math.log(nu*Math.PI)-Math.log(s)-((nu+1)/2)*Math.log1p(((x-m)/s)**2/nu);
 let out=-.5*((m1-mean)/(scale*1000))**2-.5*((m2-mean)/(scale*1000))**2;
 // Jacobian for sigma1,sigma2 and nu-1 transformations
 out+=l1+l2+ln-Math.exp(ln)/29;
 for(const x of a)out+=lpT(x,m1,s1);for(const x of b)out+=lpT(x,m2,s2);
 return out;
}
function chain(a,b,opts,seed,chainIndex){const r=rng(seed),pooled=a.concat(b),mean=pooled.reduce((s,x)=>s+x,0)/pooled.length,scale=sd(pooled),warm=opts.warm,draws=opts.draws,thin=opts.thin;let q=[a.reduce((s,x)=>s+x,0)/a.length,b.reduce((s,x)=>s+x,0)/b.length,Math.log(sd(a)),Math.log(sd(b)),Math.log(20)];let step=[scale/Math.sqrt(a.length)*.7,scale/Math.sqrt(b.length)*.7,.15,.15,.3],acc=[0,0,0,0,0],trials=[0,0,0,0,0],out=[];let lp=posterior(q,a,b,mean,scale);for(let iter=0;iter<warm+draws*thin;iter++){
 for(let k=0;k<5;k++){const proposal=q.slice();proposal[k]+=normal(r)*step[k];const next=posterior(proposal,a,b,mean,scale);trials[k]++;if(Math.log(Math.max(1e-12,r()))<next-lp){q=proposal;lp=next;acc[k]++}}
 if(iter<warm && (iter+1)%50===0){for(let k=0;k<5;k++){let rate=acc[k]/Math.max(1,trials[k]);step[k]*=Math.exp(Math.max(-.3,Math.min(.3,(rate-.35)*.7)));acc[k]=0;trials[k]=0}}
 if(iter>=warm&&(iter-warm)%thin===0){let [m1,m2,l1,l2,ln]=q,s1=Math.exp(l1),s2=Math.exp(l2);out.push([m1,m2,s1,s2,1+Math.exp(ln),m1-m2,(m1-m2)/Math.sqrt((s1*s1+s2*s2)/2),s1-s2])}
 if((iter+1)%1000===0)postMessage({type:'progress',chain:chainIndex,progress:(iter+1)/(warm+draws*thin)});
 }return out}
onmessage=e=>{try{const {a,b,opts}=e.data;if(a.length<3||b.length<3||a.concat(b).some(x=>!Number.isFinite(x)))throw Error('Servono almeno tre valori finiti per gruppo');if(sd(a)<=0||sd(b)<=0||sd(a.concat(b))<=0)throw Error('Variabilità nulla: modello non identificabile per questi dati');let chains=[];for(let c=0;c<opts.chains;c++){chains.push(chain(a,b,opts,(opts.seed+c*1000003)>>>0,c+1));postMessage({type:'progress',chain:c+1,progress:1})}postMessage({type:'done',chains})}catch(err){postMessage({type:'error',message:err.message})}};
