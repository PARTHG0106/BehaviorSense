const fs=require('node:fs');
const path=require('node:path');
const root=__dirname;
const inv=JSON.parse(fs.readFileSync(path.join(root,'source-inventory.json'),'utf8'));
const msgs=fs.readFileSync(path.join(root,'messages.jsonl'),'utf8').trim().split('\n').map(JSON.parse);
const [command,arg,extra]=process.argv.slice(2);
function header(m){return `[${m.id}] ${m.timestamp} ${m.role} ${m.kind} len=${m.text.length} copies=${m.occurrences} ${m.sources[0].source}:${m.sources[0].line}`;}
if(command==='stats') {
 console.log(JSON.stringify(inv.stats,null,2));
 console.log('Unique text hashes',new Set(msgs.map(m=>m.hash)).size);
 console.log('Claude-only messages',msgs.filter(m=>m.sources.some(s=>s.format==='claude')).length);
 console.log('Source-format totals',inv.sources.reduce((a,s)=>{const x=a[s.format]??={files:0,textMessages:0,newMessages:0};x.files++;x.textMessages+=s.textMessages;x.newMessages+=s.newMessages;return a;},{}));
 for(const s of inv.sources.filter(s=>s.format==='codex')) console.log(s.id, path.basename(s.file), JSON.stringify(s.meta), s.textMessages, s.newMessages, JSON.stringify(s.firstUser.slice(0,100)));
} else if(command==='read') {
 for(const m of msgs.filter(m=>arg.split(',').includes(m.id))) console.log(header(m)+'\n'+m.text+'\n');
} else if(command==='outline') {
 for(const m of msgs.filter(m=>(!arg||m.timestamp.startsWith(arg))&&(!extra||m.sources.some(s=>s.format===extra)))) console.log(header(m)+' '+m.text.replace(/\s+/g,' ').slice(0,260));
} else if(command==='search') {
 const regex=new RegExp(arg,'i');
 for(const m of msgs.filter(m=>(!extra||m.timestamp.startsWith(extra))&&regex.test(m.text))) {
  const at=m.text.search(regex);console.log(header(m)+'\n'+m.text.slice(Math.max(0,at-150),Math.min(m.text.length,at+1400))+'\n');
 }
} else if(command==='samples') {
 for(const s of inv.sources.filter(s=>s.format==='codex')) {
  console.log('\nSOURCE',s.id,path.basename(s.file));
  for(const m of msgs.filter(m=>m.sources[0].source===s.id).slice(0,8))console.log(header(m)+' '+JSON.stringify(m.text.slice(0,400)));
 }
} else if(command==='largest') {
 for(const m of [...msgs].sort((a,b)=>b.text.length-a.text.length).slice(0,25))console.log(header(m)+' '+m.text.replace(/\s+/g,' ').slice(0,140));
} else if(command==='index') {
 const seen=new Set();
 for(const m of msgs){
  if(seen.has(m.hash))continue;seen.add(m.hash);
  if(m.kind!=='conversation'&&m.kind!=='subagent')continue;
  if(arg&&!new RegExp(arg).test(m.timestamp))continue;
  if(m.text.startsWith('<task-notification>'))continue;
  if(m.text.startsWith('API Error:')||m.text.startsWith('<think>'))continue;
  if(extra&&m.role!==extra)continue;
  if(m.role==='assistant'&&m.text.length<900)continue;
  if(m.role==='user'&&m.text.length<30)continue;
  console.log(header(m)+'\n'+m.text.replace(/\s+/g,' ').slice(0,250)+(m.text.length>650?' ... '+m.text.replace(/\s+/g,' ').slice(-350):'')+'\n');
 }
} else if(command==='day'||command==='reading-stats') {
 const seen=new Set();const totals={};
 for(const m of msgs){
  if(seen.has(m.hash))continue;seen.add(m.hash);
  if(m.kind!=='conversation'&&m.kind!=='subagent')continue;
  if((!arg||m.timestamp.startsWith(arg))&&(!extra||m.role===extra)){
   let content=m.text;
   if(m.role==='user'&&(/Time\s+#\s+Log Message/.test(content)||/Requirement already satisfied:/.test(content))) {
    const lines=content.split('\n');
    content=lines.filter(l=>!/(Requirement already satisfied:|\bCollecting |Downloading .*whl|Using cached .*whl|Installing collected packages:|Successfully installed |\|[█▏▎▍▌▋▊▉ ]+\||━━━━━━━━|^\s*\d+(?:\.\d+)?s\s+\d+\s+\s*[╸━]|\d+%.*\[\d\d:)/.test(l)).join('\n');
   }
   totals[m.timestamp.slice(0,10)]=(totals[m.timestamp.slice(0,10)]||0)+content.length;
   if(command==='day')console.log(header(m)+'\n'+content+'\n');
  }
 }
 if(command==='reading-stats')console.log(totals);
}
