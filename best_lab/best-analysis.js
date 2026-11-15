'use strict';
const BEST={
 mean:a=>a.reduce((s,x)=>s+x,0)/a.length,
 sd(a){let m=this.mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1))},
 hdi(a,mass=.95){const x=a.slice().sort((a,b)=>a-b),n=x.length,k=Math.max(1,Math.ceil(mass*n));let best=Infinity,lo=x[0],hi=x[n-1];for(let i=0;i+k-1<n;i++){const w=x[i+k-1]-x[i];if(w<best){best=w;lo=x[i];hi=x[i+k-1]}}return [lo,hi]},
 summary(chains,index,rope,mass){let a=chains.flat().map(x=>x[index]),h=this.hdi(a,mass),inside=a.filter(x=>x>=rope[0]&&x<=rope[1]).length/a.length;return {mean:this.mean(a),median:a.slice().sort((x,y)=>x-y)[Math.floor(a.length/2)],hdi:h,inside,positive:a.filter(x=>x>0).length/a.length,rhat:this.rhat(chains.map(c=>c.map(x=>x[index]))),ess:this.ess(chains.map(c=>c.map(x=>x[index]))) }},
 // Split-Rhat, classical between/within variance diagnostic
 rhat(chains){const halves=chains.flatMap(c=>[c.slice(0,Math.floor(c.length/2)),c.slice(c.length-Math.floor(c.length/2))]);const n=halves[0].length,m=halves.length;if(n<4)return NaN;const means=halves.map(c=>this.mean(c)),W=this.mean(halves.map(c=>this.sd(c)**2)),B=n*this.sd(means)**2;return Math.sqrt(((n-1)/n*W+B/n)/W)},
 // Initial-positive-sequence estimate using pooled within-chain autocorrelation
 ess(chains){const m=chains.length,n=chains[0].length,total=m*n;const means=chains.map(c=>this.mean(c)),center=chains.map((c,i)=>c.map(x=>x-means[i]));const var0=center.reduce((s,c)=>s+c.reduce((t,x)=>t+x*x,0),0)/total;if(var0<=0)return NaN;let sum=0,prev=Infinity;const maxLag=Math.min(n-1,500);for(let lag=1;lag<maxLag;lag+=2){const ac=k=>center.reduce((s,c)=>{let v=0;for(let j=0;j<n-k;j++)v+=c[j]*c[j+k];return s+v},0)/(m*(n-k)*var0);let pair=ac(lag)+(lag+1<maxLag?ac(lag+1):0);if(pair<=0)break;pair=Math.min(pair,prev);sum+=pair;prev=pair}return Math.min(total,total/(1+2*sum))},
 parse(s){const x=s.trim().split(/[\s;]+/).filter(Boolean).map(Number);if(!x.length||x.some(v=>!Number.isFinite(v)))throw Error('Dati non validi. Usa numeri separati da spazi o righe, con il punto decimale.');return x},
 // Student-t density for predictive simulation; shape gamma via Marsaglia-Tsang
 normal(r){return Math.sqrt(-2*Math.log(Math.max(r(),1e-12)))*Math.cos(2*Math.PI*r())},
 gamma(shape,r){if(shape<1)return this.gamma(shape+1,r)*Math.pow(r(),1/shape);const d=shape-1/3,c=1/Math.sqrt(9*d);for(;;){const x=this.normal(r),v=(1+c*x)**3;if(v<=0)continue;const u=r();if(u<1-.0331*x**4||Math.log(u)<.5*x*x+d*(1-v+Math.log(v)))return d*v}},
 tSample(mu,sigma,nu,r){return mu+sigma*this.normal(r)/Math.sqrt(this.gamma(nu/2,r)*2/nu)}
};
