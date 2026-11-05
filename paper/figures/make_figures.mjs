import fs from 'node:fs/promises';
import { importRuntimeModule } from '/opt/codex/runtimes/codex-primary-runtime/plugins/openai-primary-runtime/plugins/presentations/skills/presentations/container_tools/runtime_helpers.mjs';
const { Presentation, PresentationFile } = await importRuntimeModule('@oai/artifact-tool');

const OUT = new URL('.', import.meta.url).pathname;
const W = 1600, H = 900;
const C = {
  ink: '#0F172A',
  muted: '#64748B',
  lightMuted: '#94A3B8',
  blue: '#3B82F6',
  blueDark: '#2F5DA8',
  blueLt: '#E8F1FD',
  teal: '#2A9D8F',
  tealLt: '#E5F4F2',
  green: '#59A14F',
  greenLt: '#EAF5E8',
  orange: '#F97316',
  orangeLt: '#FFF0E6',
  coral: '#E07A5F',
  coralLt: '#FBE8E3',
  yellowLt: '#FFF8DB',
  gray: '#E2E8F0',
  gray2: '#F1F5F9',
  gray3: '#CBD5E1',
  white: '#FFFFFF',
  red: '#D9485F',
};
const NO_LINE = { style: 'solid', fill: 'none', width: 0 };
const LINE = (color=C.ink, width=1.5, style='solid') => ({ style, fill: color, width });

function addShape(slide, geometry, x, y, w, h, fill='none', line=NO_LINE, name, borderRadius) {
  const cfg = { geometry, position: { left:x, top:y, width:w, height:h }, fill, line };
  if (name) cfg.name = name;
  if (borderRadius) cfg.borderRadius = borderRadius;
  return slide.shapes.add(cfg);
}
function addText(slide, text, x, y, w, h, style={}, name) {
  const s = addShape(slide, 'textbox', x, y, w, h, 'none', NO_LINE, name);
  s.text = text;
  s.text.style = {
    typeface: 'Arial',
    fontSize: '16pt',
    color: C.ink,
    alignment: 'left',
    verticalAlignment: 'middle',
    autoFit: 'shrinkText',
    wrap: 'square',
    insets: 0,
    ...style,
  };
  return s;
}
function addLabel(slide, text, x, y, w, h, color=C.muted, size='11pt', align='left', name) {
  return addText(slide, text, x, y, w, h, {fontSize:size, color, alignment:align}, name);
}
function addPanel(slide, x, y, w, h, fill=C.white, lineColor=C.gray3, radius='rounded-xl', name) {
  return addShape(slide, 'roundRect', x, y, w, h, fill, LINE(lineColor, 1.4), name, radius);
}
function arrow(slide, x, y, w, h, color=C.ink, width=2.2, name) {
  return addShape(slide, 'connector', x, y, w, h, 'none', LINE(color, width), name, undefined, {tail:{type:'arrow', width:'sm', length:'sm'}});
}
function connector(slide, a, b, opts={}) {
  return slide.shapes.connect(a, b, {
    kind: opts.kind || 'straight',
    fromSide: opts.fromSide,
    toSide: opts.toSide,
    line: LINE(opts.color || C.ink, opts.width || 2, opts.style || 'solid'),
    head: opts.head === false ? {type:'none'} : {type:'arrow', width:'sm', length:'sm'},
  });
}
function dot(slide, x, y, r, fill, lineColor=fill, name) {
  return addShape(slide, 'ellipse', x-r, y-r, 2*r, 2*r, fill, LINE(lineColor, 0.8), name);
}
function grid(slide, x, y, cols, rows, cell, gap, fills, stroke=C.white, namePrefix='grid') {
  const shapes = [];
  for (let r=0; r<rows; r++) for (let c=0; c<cols; c++) {
    const fill = Array.isArray(fills) ? fills[(r*cols+c)%fills.length] : fills;
    shapes.push(addShape(slide, 'rect', x+c*(cell+gap), y+r*(cell+gap), cell, cell, fill, LINE(stroke, 0.7), `${namePrefix}-${r}-${c}`));
  }
  return shapes;
}
function title(slide, kicker, headline, takeaway) {
  addLabel(slide, kicker.toUpperCase(), 64, 42, 300, 24, C.blueDark, '10pt', 'left');
  addText(slide, headline, 64, 66, 1150, 58, {fontSize:'28pt', bold:true, color:C.ink}, 'headline');
  if (takeaway) addLabel(slide, takeaway, 64, 128, 1180, 28, C.muted, '13pt', 'left', 'takeaway');
  addShape(slide, 'line', 64, 171, 1472, 0, 'none', LINE(C.gray3, 1.2), 'header-rule');
}
function footer(slide, text) {
  addLabel(slide, text, 64, 852, 1472, 20, C.lightMuted, '9pt', 'left', 'source-note');
}
function tag(slide, text, x, y, w, fill, color=C.ink, name) {
  addPanel(slide, x, y, w, 28, fill, fill, 'rounded-md', name);
  addText(slide, text, x+8, y+1, w-16, 26, {fontSize:'10pt', bold:true, color, alignment:'center'}, `${name||text}-label`);
}

function makeFigure1(p) {
  const s = p.slides.add(); s.background.fill = C.white;
  title(s, 'Figure 1 | Overview', 'Multimodal drafter training', 'Long visual context, few answer anchors');
  const y=206, h=590, gap=24, x1=64, w1=430, x2=x1+w1+gap, w2=504, x3=x2+w2+gap, w3=514;
  addPanel(s,x1,y,w1,h,C.blueLt,C.blue,'rounded-2xl','fig1-left');
  addPanel(s,x2,y,w2,h,C.tealLt,C.teal,'rounded-2xl','fig1-middle');
  addPanel(s,x3,y,w3,h,C.yellowLt,C.orange,'rounded-2xl','fig1-right');
  addText(s,'01',x1+24,y+18,34,28,{fontSize:'15pt',bold:true,color:C.blueDark},'f1-left-num');
  addText(s,'Long context',x1+66,y+18,310,30,{fontSize:'18pt',bold:true,color:C.ink},'f1-left-title');
  addLabel(s,'few answer anchors',x1+66,y+51,300,24,C.blueDark,'11pt','left','f1-left-sub');
  // Visual grid + text/question strip
  grid(s,x1+38,y+112,6,4,38,6,[C.blue,C.blue,C.blueLt,C.blue,C.blueLt],C.white,'f1-context');
  addLabel(s,'image-conditioned context',x1+38,y+300,270,22,C.muted,'10pt','left','f1-context-label');
  addShape(s,'rect',x1+38,y+346,324,26,C.white,LINE(C.blue,1.5),'f1-question-strip');
  addLabel(s,'question + image tokens',x1+50,y+347,300,23,C.blueDark,'10pt','left','f1-question-label');
  addLabel(s,'C',x1+382,y+137,34,50,C.blueDark,'24pt','center','f1-C');
  addLabel(s,'≫',x1+382,y+228,34,40,C.muted,'20pt','center','f1-gt');
  addLabel(s,'A',x1+382,y+275,34,50,C.orange,'24pt','center','f1-A');
  // anchor lane
  addLabel(s,'valid anchors',x1+38,y+410,160,22,C.orange,'10pt','left','f1-anchor-label');
  for (let i=0;i<4;i++) addShape(s,'roundRect',x1+38+i*76,y+446,58,38,C.orangeLt,LINE(C.orange,1.4),`f1-anchor-${i}`,'rounded-sm');
  addLabel(s,'answer positions',x1+38,y+495,200,22,C.muted,'10pt','left','f1-answer-label');
  // small padding legend
  addShape(s,'rect',x1+38,y+531,16,16,C.gray,LINE(C.gray3,0.7),'f1-pad-swatch');
  addLabel(s,'padding stays out of the objective',x1+64,y+528,320,22,C.muted,'9pt','left','f1-pad-label');

  addText(s,'02',x2+24,y+18,34,28,{fontSize:'15pt',bold:true,color:C.teal},'f1-mid-num');
  addText(s,'Capture once',x2+66,y+18,300,30,{fontSize:'18pt',bold:true,color:C.ink},'f1-mid-title');
  addLabel(s,'reuse the image-conditioned record',x2+66,y+51,392,24,C.teal,'11pt','left','f1-mid-sub');
  // source to frozen target
  const src = addPanel(s,x2+34,y+105,126,100,C.white,C.teal,'rounded-lg','f1-source');
  grid(s,x2+48,y+122,4,3,14,3,[C.blue,C.blueLt,C.blue],C.white,'f1-source-grid');
  addLabel(s,'image',x2+48,y+179,96,18,C.muted,'9pt','center','f1-source-label');
  const q = addShape(s,'roundRect',x2+186,y+128,110,46,C.white,LINE(C.teal,1.2),'f1-q','rounded-sm');
  addLabel(s,'question',x2+194,y+142,94,18,C.teal,'9pt','center','f1-q-label');
  const target = addPanel(s,x2+323,y+105,146,100,C.white,C.teal,'rounded-lg','f1-target');
  addText(s,'🔒',x2+338,y+117,28,30,{fontSize:'16pt',alignment:'center'},'f1-lock');
  addText(s,'Frozen target',x2+370,y+123,88,24,{fontSize:'10pt',bold:true,color:C.ink,alignment:'center'},'f1-target-title');
  addLabel(s,'full image',x2+348,y+157,96,20,C.teal,'9pt','center','f1-target-sub');
  connector(s, src, q, {fromSide:'right',toSide:'left',color:C.teal,width:1.6});
  connector(s, q, target, {fromSide:'right',toSide:'left',color:C.teal,width:1.6});
  // record
  addLabel(s,'image-conditioned record',x2+34,y+238,250,22,C.teal,'10pt','left','f1-record-label');
  const record = addPanel(s,x2+34,y+270,214,145,C.white,C.teal,'rounded-lg','f1-record');
  grid(s,x2+50,y+291,5,4,20,4,[C.blue,C.blueLt,C.blue,C.blueLt,C.blue],C.white,'f1-record-grid');
  for (const [dx,dy] of [[70,302],[113,342],[155,322],[195,382]]) dot(s,x2+dx,y+dy,4,C.orange,C.orange,'f1-record-dot');
  addLabel(s,'spatial index',x2+49,y+384,160,18,C.teal,'9pt','left','f1-index-label');
  // grouping
  const group = addPanel(s,x2+282,y+270,186,145,C.white,C.teal,'rounded-lg','f1-group');
  addLabel(s,'C × A',x2+310,y+298,132,34,C.ink,'22pt','center','f1-group-eq');
  addLabel(s,'group by context\nand anchor count',x2+301,y+347,150,42,C.muted,'9pt','center','f1-group-label');
  connector(s, target, record, {fromSide:'bottom',toSide:'top',color:C.teal,width:1.8,kind:'elbow'});
  connector(s, record, group, {fromSide:'right',toSide:'left',color:C.teal,width:1.8});
  // reuse strip
  addLabel(s,'one target pass',x2+34,y+458,150,20,C.teal,'10pt','left','f1-once-label');
  addLabel(s,'many drafter epochs',x2+278,y+458,180,20,C.teal,'10pt','left','f1-many-label');
  const one = addShape(s,'roundRect',x2+34,y+486,94,38,C.teal,LINE(C.teal,1.2),'f1-one','rounded-sm');
  addText(s,'1× target',x2+38,y+489,86,30,{fontSize:'10pt',bold:true,color:C.white,alignment:'center'},'f1-one-text');
  for (let i=0;i<3;i++) {
    const e=addShape(s,'roundRect',x2+278+i*62,y+486,50,38,C.white,LINE(C.teal,1.2),`f1-epoch-${i}`,'rounded-sm');
    addText(s,`E${i+1}`,x2+284+i*62,y+489,38,30,{fontSize:'10pt',bold:true,color:C.teal,alignment:'center'},`f1-epoch-${i}-text`);
  }
  connector(s, one, addShape(s,'roundRect',x2+278,y+486,50,38,'none',NO_LINE,'f1-epoch-anchor','rounded-sm'), {fromSide:'right',toSide:'left',color:C.teal,width:1.5});
  addLabel(s,'record reused',x2+34,y+542,140,18,C.muted,'9pt','left','f1-reuse-label');

  addText(s,'03',x3+24,y+18,34,28,{fontSize:'15pt',bold:true,color:C.orange},'f1-right-num');
  addText(s,'Score by objective',x3+66,y+18,390,30,{fontSize:'18pt',bold:true,color:C.ink},'f1-right-title');
  addLabel(s,'one drafter pass, two selection units',x3+66,y+51,410,24,C.orange,'11pt','left','f1-right-sub');
  const rec2 = addPanel(s,x3+34,y+115,126,92,C.white,C.teal,'rounded-lg','f1-right-record');
  grid(s,x3+48,y+130,4,3,14,3,[C.blue,C.blueLt,C.blue],C.white,'f1-right-grid');
  addLabel(s,'record',x3+47,y+184,90,18,C.teal,'9pt','center','f1-right-record-label');
  const drafter = addPanel(s,x3+194,y+115,160,92,C.white,C.orange,'rounded-lg','f1-drafter');
  addText(s,'Drafter',x3+212,y+136,124,28,{fontSize:'15pt',bold:true,color:C.ink,alignment:'center'},'f1-drafter-title');
  addLabel(s,'parallel block',x3+211,y+171,124,18,C.orange,'9pt','center','f1-drafter-sub');
  connector(s, rec2, drafter, {fromSide:'right',toSide:'left',color:C.ink,width:2});
  const row = addPanel(s,x3+34,y+285,202,208,C.white,C.blue,'rounded-lg','f1-row');
  tag(s,'Row-local',x3+50,y+304,100,C.blueLt,C.blueDark,'f1-row-tag');
  addLabel(s,'score selected rows',x3+50,y+348,162,20,C.muted,'9pt','left','f1-row-desc');
  for (let i=0;i<5;i++) {
    const fill = i<2 ? C.orangeLt : C.gray2;
    const col = i<2 ? C.orange : C.gray3;
    addShape(s,'roundRect',x3+54,y+387+i*21,154,14,fill,LINE(col,0.8),`f1-row-${i}`,'rounded-sm');
  }
  addLabel(s,'only objective-dependent rows',x3+50,y+465,160,18,C.blueDark,'9pt','left','f1-row-foot');
  const pref = addPanel(s,x3+274,y+285,202,208,C.white,C.green,'rounded-lg','f1-pref');
  tag(s,'Prefix-coupled',x3+290,y+304,132,C.greenLt,C.green,'f1-pref-tag');
  addLabel(s,'keep each valid block',x3+290,y+348,162,20,C.muted,'9pt','left','f1-pref-desc');
  for (let i=0;i<5;i++) {
    const fill = i<4 ? C.greenLt : C.gray2;
    const col = i<4 ? C.green : C.gray3;
    addShape(s,'roundRect',x3+294,y+387+i*21,154,14,fill,LINE(col,0.8),`f1-pref-${i}`,'rounded-sm');
  }
  addLabel(s,'preserve within-block dependency',x3+290,y+465,170,18,C.green,'9pt','left','f1-pref-foot');
  connector(s,drafter,row,{fromSide:'bottom',toSide:'top',color:C.blueDark,width:1.8,kind:'elbow'});
  connector(s,drafter,pref,{fromSide:'bottom',toSide:'top',color:C.green,width:1.8,kind:'elbow'});
  addShape(s,'line',x3+34,y+528,442,0,'none',LINE(C.orange,1.4),'f1-boundary');
  addLabel(s,'Target still reads the full image',x3+34,y+550,430,24,C.ink,'11pt','center','f1-target-boundary');
  addLabel(s,'optimization lives in record reuse and drafter packing',x3+34,y+568,430,20,C.muted,'9pt','center','f1-target-boundary-sub');
  footer(s,'VelaDraft overview. The full image-conditioned record is captured once; the drafter reuses it and scores only what each objective depends on.');
}

function makeFigure2(p) {
  const s = p.slides.add(); s.background.fill = C.white;
  title(s, 'Figure 2 | Visual evidence', 'Visual rows carry proposal signal', 'Wrong-image replacement hurts more than compacting the visual grid');
  const y=206;
  addPanel(s,64,y,430,590,C.gray2,C.gray3,'rounded-2xl','fig2-left');
  addPanel(s,520,y,1016,590,C.white,C.gray3,'rounded-2xl','fig2-right');
  addText(s,'Visual views',92,y+24,220,32,{fontSize:'18pt',bold:true,color:C.ink},'f2-left-title');
  addLabel(s,'same target trajectory, different drafter rows',92,y+56,320,22,C.muted,'10pt','left','f2-left-sub');
  const views = [
    ['Full',C.blue,C.blueLt,'full'],
    ['Quarter-grid',C.teal,C.tealLt,'quarter'],
    ['No visual rows',C.gray3,C.white,'none'],
    ['Shuffled',C.orange,C.orangeLt,'shuffled'],
    ['Cross-image',C.red,C.coralLt,'cross'],
  ];
  views.forEach(([label,border,fill,key],i)=>{
    const yy=y+112+i*84;
    addPanel(s,92,yy,366,62,C.white,border,'rounded-lg',`f2-view-${key}`);
    // tiny grid icon
    const fills = key==='none' ? [C.gray2] : key==='cross' ? [C.coralLt,C.blueLt] : key==='quarter' ? [C.tealLt,C.white] : key==='shuffled' ? [C.orangeLt,C.blueLt] : [C.blueLt,C.blue];
    grid(s,112,yy+15,4,3,9,3,fills,C.white,`f2-icon-${key}`);
    addText(s,label,192,yy+12,190,26,{fontSize:'13pt',bold:true,color:C.ink},`f2-view-label-${key}`);
    if (key==='cross') addLabel(s,'wrong image',192,yy+37,170,18,C.red,'9pt','left',`f2-view-note-${key}`);
    else if (key==='none') addLabel(s,'text positions only',192,yy+37,170,18,C.muted,'9pt','left',`f2-view-note-${key}`);
    else addLabel(s,key==='full'?'reference':key==='quarter'?'1 of 4 visual cells':'same rows, altered order',192,yy+37,220,18,C.muted,'9pt','left',`f2-view-note-${key}`);
  });
  addShape(s,'line',92,y+536,366,0,'none',LINE(C.gray3,1.2),'f2-left-rule');
  addLabel(s,'Full = 0 pp reference',92,y+552,250,20,C.blueDark,'10pt','left','f2-baseline');
  // chart
  addText(s,'Fixed-trajectory top-1 change vs Full (pp)',560,y+24,600,34,{fontSize:'17pt',bold:true,color:C.ink},'f2-chart-title');
  addLabel(s,'negative values indicate fewer matched proposal tokens',560,y+60,590,20,C.muted,'10pt','left','f2-chart-sub');
  const plotX=780, plotY=y+116, plotW=650, rowH=78, min=-7.5, max=1.5;
  const scale = plotW/(max-min);
  const x0 = plotX + (0-min)*scale;
  addShape(s,'line',x0,plotY-18,0,420,'none',LINE(C.ink,2.2),'f2-zero');
  // tick marks and labels
  [-7,-5,-3,-1,0,1].forEach(v=>{
    const xx=plotX+(v-min)*scale;
    addShape(s,'line',xx,plotY+392,0,10,'none',LINE(C.gray3,1),'f2-tick-'+v);
    addLabel(s,String(v),xx-18,plotY+406,36,18,C.muted,'9pt','center','f2-tick-label-'+v);
  });
  addLabel(s,'DFlash2',560,plotY-5,112,20,C.blueDark,'10pt','left','f2-legend-df2');
  addShape(s,'roundRect',670,plotY+0,22,14,C.blueDark,LINE(C.blueDark,0.5),'f2-legend-box1','rounded-sm');
  addLabel(s,'DFly',560,plotY+24,112,20,C.coral,'10pt','left','f2-legend-dfly');
  addShape(s,'roundRect',670,plotY+29,22,14,C.coral,LINE(C.coral,0.5),'f2-legend-box2','rounded-sm');
  const vals = [
    ['Text-only',-4.58,-4.07],
    ['Grid-quarter',-1.66,-1.49],
    ['Shuffled',0.07,-0.01],
    ['Cross-image',-6.47,-6.36],
  ];
  vals.forEach(([lab,a,b],i)=>{
    const yy=plotY+80+i*rowH;
    addText(s,lab,560,yy-12,180,26,{fontSize:'12pt',bold:true,color:C.ink},`f2-row-label-${i}`);
    addLabel(s,i===3?'wrong image':i===1?'compact visual grid':'',560,yy+16,170,18,i===3?C.red:C.muted,'9pt','left',`f2-row-note-${i}`);
    const drawBar=(v,offset,color,txt,name)=>{
      const x=plotX+(v-min)*scale;
      const left=Math.min(x,x0), width=Math.max(Math.abs(x-x0),2);
      addShape(s,'roundRect',left,yy+offset,width,14,color,LINE(color,0.5),name,'rounded-sm');
      const tx=v<0?left-58:x+8;
      addLabel(s,txt,tx,yy+offset-4,52,22,color,'10pt',v<0?'right':'left',name+'-label');
    };
    drawBar(a,0,C.blueDark,a.toFixed(2),`f2-bar-a-${i}`);
    drawBar(b,25,C.coral,b.toFixed(2),`f2-bar-b-${i}`);
  });
  // highlight callout
  addPanel(s,1182,y+473,300,76,C.coralLt,C.coral,'rounded-lg','f2-callout');
  addText(s,'Wrong image',1200,y+485,145,24,{fontSize:'14pt',bold:true,color:C.red},'f2-callout-title');
  addLabel(s,'−6.36 to −6.47 pp',1200,y+515,220,22,C.red,'12pt','left','f2-callout-value');
  addLabel(s,'The drafter uses image-specific rows',560,y+536,590,22,C.ink,'10pt','left','f2-interpretation');
  addLabel(s,'A quarter-grid keeps most of the signal: −1.49 to −1.66 pp',560,y+560,650,20,C.muted,'10pt','left','f2-interpretation-sub');
  footer(s,'384 image-disjoint requests. Target reads the complete image; only visual rows exposed to the drafter change. Values are fixed-trajectory top-1 changes, not final task accuracy.');
}

function makeFigure3(p) {
  const s = p.slides.add(); s.background.fill = C.white;
  title(s, 'Figure 3 | Training cost', 'Training savings at full scale', 'Matched 75K-update runs on Qwen3-VL-8B');
  addPanel(s,64,206,1472,590,C.white,C.gray3,'rounded-2xl','fig3-main');
  addText(s,'Elapsed training time (hours)',96,236,360,30,{fontSize:'18pt',bold:true,color:C.ink},'f3-title');
  addLabel(s,'Reference and VelaDraft use the same matched recipe',96,272,420,22,C.muted,'10pt','left','f3-sub');
  // legend
  addShape(s,'roundRect',1066,238,20,14,C.gray3,LINE(C.gray3,0.5),'f3-legend-ref','rounded-sm');
  addLabel(s,'Reference',1096,233,110,22,C.muted,'10pt','left','f3-legend-ref-label');
  addShape(s,'roundRect',1230,238,20,14,C.teal,LINE(C.teal,0.5),'f3-legend-veladraft','rounded-sm');
  addLabel(s,'VelaDraft',1260,233,110,22,C.teal,'10pt','left','f3-legend-veladraft-label');
  const plotX=340, plotY=348, plotW=930, maxH=65, scale=plotW/maxH;
  // grid lines / ticks
  for (let v=0;v<=60;v+=10) {
    const xx=plotX+v*scale;
    addShape(s,'line',xx,plotY-22,0,260,'none',LINE(C.gray2,1),'f3-grid-'+v);
    addLabel(s,String(v),xx-16,plotY+254,32,18,C.muted,'9pt','center','f3-tick-'+v);
  }
  addLabel(s,'hours',plotX+plotW-44,plotY+278,60,18,C.muted,'9pt','right','f3-hours');
  const rows=[
    {name:'DFlash2',ref:196371/3600,opt:179051/3600,saved:'−4.81 h',color:C.blueDark},
    {name:'DFly',ref:225894/3600,opt:193820/3600,saved:'−8.91 h',color:C.coral},
  ];
  rows.forEach((d,i)=>{
    const yy=plotY+i*112;
    addText(s,d.name,96,yy-4,190,34,{fontSize:'18pt',bold:true,color:C.ink},'f3-row-name-'+i);
    addLabel(s,'75K updates',96,yy+34,160,18,C.muted,'9pt','left','f3-row-sub-'+i);
    const refW=d.ref*scale, optW=d.opt*scale;
    addShape(s,'roundRect',plotX,yy,refW,28,C.gray3,LINE(C.gray3,0.5),'f3-ref-bar-'+i,'rounded-sm');
    addShape(s,'roundRect',plotX,yy+38,optW,28,C.teal,LINE(C.teal,0.5),'f3-opt-bar-'+i,'rounded-sm');
    addLabel(s,`${d.ref.toFixed(1)} h`,plotX+refW+10,yy+2,90,22,C.muted,'10pt','left','f3-ref-label-'+i);
    addLabel(s,`${d.opt.toFixed(1)} h`,plotX+optW+10,yy+40,90,22,C.teal,'10pt','left','f3-opt-label-'+i);
    addPanel(s,plotX+plotW+28,yy+3,168,58,i===0?C.blueLt:C.coralLt,i===0?C.blueDark:C.coral,'rounded-lg','f3-save-'+i);
    addText(s,`${d.saved} saved`,plotX+plotW+40,yy+13,144,30,{fontSize:'14pt',bold:true,color:i===0?C.blueDark:C.red,alignment:'center'},'f3-save-text-'+i);
  });
  addLabel(s,'reference',plotX-78,plotY+4,68,18,C.muted,'9pt','right','f3-ref-ylabel');
  addLabel(s,'VelaDraft',plotX-78,plotY+42,68,18,C.teal,'9pt','right','f3-opt-ylabel');
  // Feature reuse footer ribbon
  addPanel(s,96,620,1320,108,C.tealLt,C.teal,'rounded-xl','f3-reuse-ribbon');
  addText(s,'Feature reuse',122,640,180,26,{fontSize:'15pt',bold:true,color:C.teal},'f3-ribbon-title');
  addText(s,'576 → 96',334,636,180,38,{fontSize:'22pt',bold:true,color:C.ink,alignment:'center'},'f3-ribbon-number');
  addLabel(s,'vision-encoder calls across 96 images × 6 passes',334,680,360,20,C.muted,'9pt','center','f3-ribbon-label');
  addText(s,'480 repeated encodings removed',776,650,360,30,{fontSize:'15pt',bold:true,color:C.ink,alignment:'center'},'f3-ribbon-saving');
  addLabel(s,'same image-conditioned records are reused by later epochs',776,686,360,18,C.muted,'9pt','center','f3-ribbon-saving-sub');
  footer(s,'Wall times include training, validation, and checkpoint phases. Reference and optimized runs use matched Qwen3-VL-8B recipes on one A100 80 GB.');
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT, {recursive:true});
  const p = Presentation.create({slideSize:{width:W,height:H}});
  makeFigure1(p); makeFigure2(p); makeFigure3(p);
  const rendered = p.slides.items;
  for (let i=0;i<rendered.length;i++) {
    const png = await p.export({slide:rendered[i], format:'png', scale:2});
    await writeBlob(`${OUT}/figure-${i+1}.png`, png);
  }
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(`${OUT}/VelaDraft_paper_figures_v2.pptx`);
  const inspect = await p.inspect({select:'$.slides[*]', include:['id','index','elements.id','elements.text']});
  await fs.writeFile(`${OUT}/inspect.ndjson`, JSON.stringify(inspect,null,2));
}
main().catch(err=>{ console.error(err); process.exitCode=1; });
