const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const workspace = 'C:/Users/Admin/Desktop/EmotionSense-Extended';
const claudeRoot = 'C:/Users/Admin/.claude/projects/C--Users-Admin-Desktop-EmotionSense-Extended';
const codexRoot = 'C:/Users/Admin/.codex';
const outputRoot = path.join(workspace, 'artifacts/history-recovery');
const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const normalize = value => value.replace(/\r\n/g, '\n').trim();
const walk = root => fs.existsSync(root) ? fs.readdirSync(root, {withFileTypes:true}).flatMap(e => e.isDirectory() ? walk(path.join(root, e.name)) : [path.join(root, e.name)]) : [];
const readJsonl = file => fs.readFileSync(file, 'utf8').split(/\r?\n/).flatMap((line, i) => {
  if (!line.trim()) return [];
  try { return [{line:i+1, data:JSON.parse(line)}]; }
  catch (error) { return [{line:i+1, error:String(error)}]; }
});
function visibleText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.filter(x => ['text','input_text','output_text'].includes(x?.type)).map(x=>x.text || '').join('\n');
}
function messageKind(role, text, data={}) {
  if (data.isMeta || /^(This session is being continued|This is a continuation|<context|<environment_context>|<permissions instructions>|# AGENTS\.md|<user_instructions>|You are an AI|You are Codex)/.test(text)) return 'context';
  if (/^\[Request interrupted|^<local-command|^<command-name>|^<command-message>|^Caveat:/.test(text)) return 'control';
  return data.isSidechain ? 'subagent' : 'conversation';
}
const sources=[];
const messages=[];
const byIdentity=new Map();
const byTimestampText=new Map();
let messageOccurrences=0;
let representationsSkipped=0;
let importedToolBlocksSkipped=0;
let importedCopiesMatched=0;
const importsFile=path.join(codexRoot,'external_agent_session_imports.json');
const projectImports=fs.existsSync(importsFile)?JSON.parse(fs.readFileSync(importsFile,'utf8')).records.filter(r=>r.source_path.includes('C--Users-Admin-Desktop-EmotionSense-Extended')):[];
const importByThread=new Map(projectImports.map(r=>[r.imported_thread_id,r]));
const firstSamples=[];
function addMessage(item, source) {
  item.text=normalize(item.text);
  if (!item.text) return;
  messageOccurrences++;
  const contentHash=hash(item.role+'\0'+item.text);
  const key=item.timestamp+'\0'+contentHash;
  const identity=item.uuid ? item.uuid+'\0'+contentHash : null;
  let existing=(identity && byIdentity.get(identity)) || byTimestampText.get(key);
  const ref={file:source.file, line:item.line, source:source.id, format:source.format};
  if (existing) {
    existing.occurrences++;
    if (!existing.sources.some(s=>s.file===ref.file&&s.line===ref.line)) existing.sources.push(ref);
    source.duplicateOccurrences++;
  } else {
    existing={timestamp:item.timestamp, role:item.role, kind:item.kind, text:item.text, hash:contentHash, uuid:item.uuid||null, model:item.model||null, occurrences:1, sources:[ref]};
    messages.push(existing);
    source.newMessages++;
  }
  if (identity) byIdentity.set(identity,existing);
  byTimestampText.set(key,existing);
  source.messageHashes.push(contentHash);
}
const claudeFiles=walk(claudeRoot).filter(f=>f.endsWith('.jsonl')).sort();
for (const [i,file] of claudeFiles.entries()) {
  const source={id:'C'+String(i+1).padStart(3,'0'),format:'claude',file,bytes:fs.statSync(file).size,sha256:hash(fs.readFileSync(file)),types:{},models:[],cwd:[],sessionIds:[],firstTimestamp:null,lastTimestamp:null,firstUser:'',newMessages:0,duplicateOccurrences:0,messageHashes:[],errors:[]};
  for (const {line,data,error} of readJsonl(file)) {
    if (error) {source.errors.push({line,error});continue;}
    source.types[data.type]=(source.types[data.type]||0)+1;
    if (data.timestamp) {source.firstTimestamp = !source.firstTimestamp || data.timestamp<source.firstTimestamp ? data.timestamp : source.firstTimestamp;source.lastTimestamp = !source.lastTimestamp || data.timestamp>source.lastTimestamp ? data.timestamp : source.lastTimestamp;}
    if (data.cwd&&!source.cwd.includes(data.cwd))source.cwd.push(data.cwd);
    if (data.sessionId&&!source.sessionIds.includes(data.sessionId))source.sessionIds.push(data.sessionId);
    if (data.message?.model&&!source.models.includes(data.message.model))source.models.push(data.message.model);
    if (data.type==='custom-title')source.title=data.customTitle;
    if (!['user','assistant'].includes(data.type)||!data.message)continue;
    const text=visibleText(data.message.content);
    if (!text.trim()) continue;
    const role=data.message.role||data.type;
    if(role==='user'&&!source.firstUser)source.firstUser=text.slice(0,350);
    addMessage({timestamp:data.timestamp||'',role,text,line,uuid:data.uuid,model:data.message.model,kind:messageKind(role,text,data)},source);
    if(firstSamples.length<3)firstSamples.push({format:'claude',keys:Object.keys(data),messageKeys:Object.keys(data.message),contentTypes:Array.isArray(data.message.content)?data.message.content.map(x=>x.type):typeof data.message.content});
  }
  source.sequenceHash=hash(source.messageHashes.join('\n'));
  source.textMessages=source.messageHashes.length;
  sources.push(source);
}

const codexFiles=[...walk(path.join(codexRoot,'sessions')),...walk(path.join(codexRoot,'archived_sessions'))].filter(f=>f.endsWith('.jsonl')).sort();
const codexExcluded=[];
const claudeByText=new Map();
for(const m of messages){const key=m.hash;if(!claudeByText.has(key))claudeByText.set(key,[]);claudeByText.get(key).push(m);}
for (const [i,file] of codexFiles.entries()) {
  // Scope using local rollout metadata, without opening SQLite or provider state.
  const descriptor=fs.openSync(file,'r');
  const head=Buffer.alloc(Math.min(fs.statSync(file).size,300000));
  fs.readSync(descriptor,head,0,head.length,0);fs.closeSync(descriptor);
  const firstLine=head.toString('utf8').split('\n')[0];
  let meta;
  try { const parsed=JSON.parse(firstLine);if(parsed.type==='session_meta')meta=parsed.payload; } catch {}
  const relevant=meta && (/emotionsense-extended/i.test(meta.cwd||'')||importByThread.has(meta.id));
  if(!relevant){codexExcluded.push({file,cwd:meta?.cwd||null,id:meta?.id||null});continue;}
  const entries=readJsonl(file);
  const currentRequest=entries.some(e=>e.data?.type==='response_item'&&e.data?.payload?.role==='user'&&visibleText(e.data.payload.content).includes('Identify the Claude/Codex conversation files belonging to EmotionSense-Extended'));
  if(currentRequest){codexExcluded.push({file,cwd:meta.cwd,id:meta.id,reason:'current history-recovery conversation'});continue;}
  const source={id:'X'+String(i+1).padStart(3,'0'),format:'codex',file,bytes:fs.statSync(file).size,sha256:hash(fs.readFileSync(file)),meta:{id:meta.id,cwd:meta.cwd,timestamp:meta.timestamp,source:meta.source,originator:meta.originator,forked_from_id:meta.forked_from_id},types:{},models:[],cwd:[meta.cwd],sessionIds:[meta.id],firstTimestamp:null,lastTimestamp:null,firstUser:'',newMessages:0,duplicateOccurrences:0,messageHashes:[],errors:[]};
  const imported=importByThread.get(meta.id);
  if(imported)source.importedFrom={sourcePath:imported.source_path,importedAt:imported.imported_at,title:imported.title};
  const responseTexts=new Set(entries.filter(e=>e.data?.type==='response_item'&&e.data?.payload?.type==='message'&&['user','assistant'].includes(e.data?.payload?.role)).map(e=>e.data.payload.role+'\0'+normalize(visibleText(e.data.payload.content))));
  for (const {line,data,error} of entries) {
    if(error){source.errors.push({line,error});continue;}
    source.types[data.type]=(source.types[data.type]||0)+1;
    if(data.timestamp){source.firstTimestamp=!source.firstTimestamp||data.timestamp<source.firstTimestamp?data.timestamp:source.firstTimestamp;source.lastTimestamp=!source.lastTimestamp||data.timestamp>source.lastTimestamp?data.timestamp:source.lastTimestamp;}
    if(data.type==='turn_context'&&data.payload?.model&&!source.models.includes(data.payload.model))source.models.push(data.payload.model);
    let role,text,kind;
    if(data.type==='response_item'&&data.payload?.type==='message'&&['user','assistant'].includes(data.payload.role)) {
      if(data.payload.channel==='analysis')continue;
      role=data.payload.role;text=visibleText(data.payload.content);kind=messageKind(role,text);
    } else if(data.type==='event_msg'&&['user_message','agent_message'].includes(data.payload?.type)) {
      role=data.payload.type==='user_message'?'user':'assistant';text=data.payload.message||'';kind=messageKind(role,text);
      if(responseTexts.has(role+'\0'+normalize(text))){representationsSkipped++;continue;}
    } else continue;
    if(/^\[external_agent_tool_(call|result)/.test(text)){importedToolBlocksSkipped++;continue;}
    if(imported){
      text=text.replace(/\[external unsupported block: [^\]]+\]\s*/g,'').trim();
      const candidates=claudeByText.get(hash(role+'\0'+normalize(text)))||[];
      const originalFile=path.basename(imported.source_path);
      const existing=candidates.find(m=>m.sources.some(s=>path.basename(s.file)===originalFile))||candidates[0];
      if(existing){
        messageOccurrences++;importedCopiesMatched++;source.duplicateOccurrences++;
        source.messageHashes.push(existing.hash);
        existing.occurrences++;
        existing.sources.push({file:source.file,line,source:source.id,format:source.format,importTimestamp:data.timestamp});
        continue;
      }
    }
    if(!text.trim())continue;
    if(role==='user'&&kind==='conversation'&&!source.firstUser)source.firstUser=text.slice(0,350);
    addMessage({timestamp:data.timestamp||'',role,text,line,kind},source);
  }
  source.sequenceHash=hash(source.messageHashes.join('\n'));
  source.textMessages=source.messageHashes.length;
  sources.push(source);
}

messages.sort((a,b)=>a.timestamp.localeCompare(b.timestamp)||a.sources[0].file.localeCompare(b.sources[0].file)||a.sources[0].line-b.sources[0].line);
for(const [i,m] of messages.entries())m.id='M'+String(i+1).padStart(5,'0');
const groupBy=fn=>{const groups=new Map();for(const s of sources){const key=fn(s);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(s.id);}return [...groups.values()].filter(g=>g.length>1);};
const days={};
for(const m of messages){const day=m.timestamp.slice(0,10);const d=days[day]??={user:0,assistant:0,context:0,chars:0};d[m.role]++;if(m.kind==='context')d.context++;d.chars+=m.text.length;}
const stats={createdAt:new Date().toISOString(),claudeFiles:claudeFiles.length,codexFilesScanned:codexFiles.length,codexFilesIncluded:sources.filter(s=>s.format==='codex').length,projectImportRecords:projectImports.length,totalSourceBytes:sources.reduce((a,s)=>a+s.bytes,0),messageOccurrences,representationsSkipped,importedToolBlocksSkipped,importedCopiesMatched,uniqueMessages:messages.length,uniqueChars:messages.reduce((a,m)=>a+m.text.length,0),firstTimestamp:messages[0]?.timestamp,lastTimestamp:messages.at(-1)?.timestamp,parseErrors:sources.flatMap(s=>s.errors.map(e=>({file:s.file,...e}))),days};
fs.mkdirSync(outputRoot,{recursive:true});
fs.writeFileSync(path.join(outputRoot,'source-inventory.json'),JSON.stringify({stats,sources,exactFileDuplicateGroups:groupBy(s=>s.sha256),sameTextSequenceGroups:groupBy(s=>s.sequenceHash),projectImports,codexExcluded},null,2));
fs.writeFileSync(path.join(outputRoot,'messages.jsonl'),messages.map(m=>JSON.stringify(m)).join('\n')+'\n');
for(const day of Object.keys(days)) {
  fs.writeFileSync(path.join(outputRoot,day+'.txt'),messages.filter(m=>m.timestamp.startsWith(day)).map(m=>`\n[${m.id}] ${m.timestamp} ${m.role.toUpperCase()} (${m.kind}) [${m.sources[0].source}:${m.sources[0].line}; copies=${m.occurrences}]\n${m.text}\n`).join('\n'));
}
console.log(JSON.stringify({stats,sameTextSequenceGroupCount:groupBy(s=>s.sequenceHash).length},null,2));
