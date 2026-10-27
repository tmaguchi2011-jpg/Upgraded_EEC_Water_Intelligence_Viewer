"""High-recall IID East Highline PDF extraction.
OpenCV+Tesseract local pass; optional OpenAI vision table pass.
All outputs are unverified candidates. Never use OCR/AI as a source of truth.
"""
import base64, io, json, os, re
from datetime import datetime
import cv2
import fitz
import numpy as np
from PIL import Image
import pytesseract

PARAMETER_ALIASES = {
 'aluminum':'Aluminum','iron':'Iron','turbidity':'Turbidity','specific conductance':'Conductivity',
 'electrical conductivity':'Conductivity','conductivity':'Conductivity',
 'total filterable residue':'TDS','total dissolved solids':'TDS','tds':'TDS',
 'sulfate':'Sulfate','chloride':'Chloride','ph (lab)':'pH','ph (field)':'pH',
 'alkalinity, total':'Alkalinity','alkalinity':'Alkalinity',
 'calcium':'Calcium','magnesium':'Magnesium','sodium':'Sodium','reactive silica':'Silica',
 'silica':'Silica','total suspended solids':'TSS','apparent color':'Color',
 'manganese':'Manganese','potassium':'Potassium','boron':'Boron',
 'barium':'Barium','nitrate':'Nitrate','fluoride':'Fluoride','hardness, total':'Hardness',
 'bicarbonate':'Bicarbonate',
}
ORDERED = sorted(PARAMETER_ALIASES, key=len, reverse=True)
RE_OTHER = re.compile(r'westside\s+main|central\s+main|all[- ]american\s+canal|aac\s+drop|drop\s*4',re.I)
RE_EHL = re.compile(r'east\s*high\s*(?:line|lane)(?:\s*canal)?',re.I)
RE_YEAR = re.compile(r'(20\d{2})')
RE_NUM = re.compile(r'^(?:[<>]\s*)?\d+(?:,\d{3})*(?:\.\d+)?$|^(?:ND|N/D|NOT DETECTED)$', re.I)


def render(page, dpi=230):
    pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False)
    arr=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,3)
    return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)


def preprocess(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.fastNlMeansDenoising(gray,h=6)
    gray=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    # Keep both grayscale and adaptive binarization: colors may contain the important results.
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,35,13)
    return binary


def ocr_rows(image):
    """OCR by spatial row, rather than fixed-width splitting of page text."""
    d=pytesseract.image_to_data(image,config='--oem 3 --psm 6',output_type=pytesseract.Output.DICT)
    lines={}
    for i,t in enumerate(d['text']):
        t=t.strip()
        if not t:continue
        key=(d['block_num'][i],d['par_num'][i],d['line_num'][i])
        lines.setdefault(key,[]).append((d['left'][i],t))
    return [' '.join(t for _,t in sorted(v)) for v in lines.values()]


def pname(raw):
    raw=re.sub(r'\s+',' ',raw.lower()).strip()
    for alias in ORDERED:
        if raw.startswith(alias):return PARAMETER_ALIASES[alias]
    return None


def normalize_units(parameter,units,value_text):
    u=(units or '').lower().replace('µ','u').replace('μ','u').replace(' ','')
    if parameter in ('Aluminum','Iron','Manganese','Barium','Boron'):
        if 'ug/l' in u:return 'mg/L',0.001
        if 'mg/l' in u:return 'mg/L',1.0
        return None,None
    if parameter=='Conductivity':
        if 'umhos/cm' in u or 'us/cm' in u:return 'uS/cm',1.0
        return None,None
    if parameter=='Turbidity':return ('NTU',1.) if 'ntu' in u else (None,None)
    if parameter=='pH':return 'pH',1.
    if parameter=='Color':return 'color units',1.
    if parameter=='Alkalinity':return ('mg/L as CaCO3',1.) if 'mg/l' in u else (None,None)
    if parameter=='Silica':
        if 'caco3' in u:return 'mg/L as CaCO3',1.
        if 'sio2' in u:return 'mg/L as SiO2',1.
        return None,None  # refuse chemically ambiguous conversion
    return ('mg/L',1.) if 'mg/l' in u else (None,None)


def local_candidates(lines,page_index,year,date,source):
    out=[]
    for line in lines:
        name=pname(line)
        if not name:continue
        # Match standard table: parameter, analytical method, result, [units or reporting limit], units.
        # Method names and numerical concentration must not be confused.
        m=re.search(r'\b(?:EPA\s*\d+(?:\.\d+)?(?:-M)?|SM\s*\d+[A-Z]*|Field)\s+(ND|N/D|<\s*\d+(?:\.\d+)?|\d+(?:,\d{3})*(?:\.\d+)?)\b',line,re.I)
        if not m:continue
        raw=m.group(1).replace(' ','')
        suffix=line[m.end():]
        um=re.search(r'(ug\s*/\s*l|µg\s*/\s*l|mg\s*/\s*l|NTU|umhos\s*/\s*cm|uS\s*/\s*cm|pH\s*Units|Color\s*Units)(?:\s+as\s+(?:CaCO3|SiO2))?', suffix, re.I)
        if not um:continue
        unit=um.group(0)
        normalized,factor=normalize_units(name,unit,raw)
        if normalized is None:continue
        censored=raw.upper().startswith(('ND','N/D','<'))
        try:v=None if censored else float(raw.replace(',',''))*factor
        except ValueError:continue
        if v is not None and ((name=='pH' and not 0<=v<=14) or (name=='Turbidity' and v<0)):
            continue
        out.append(dict(sample_date=date,year=year,parameter=name,value=v,unit=normalized,
           reported_result=raw,method='CV/Tesseract',source_page=page_index,raw_line=line[:700],
           status='needs_review',confidence_note='OpenCV/Tesseract candidate: check station, date, value and unit against original PDF'))
    return out

SCHEMA = {
 'name':'iid_water_rows','strict':True,
 'schema':{'type':'object','properties':{
  'station':{'type':'string'},'sample_date':{'type':['string','null']},
  'rows':{'type':'array','items':{'type':'object','properties':{
   'parameter':{'type':'string'},'reported_result':{'type':'string'},
   'unit':{'type':'string'},'method':{'type':'string'},'source_row':{'type':'string'}},
   'required':['parameter','reported_result','unit','method','source_row'],'additionalProperties':False}}},
  'required':['station','sample_date','rows'],'additionalProperties':False}
}


def ai_candidates(img,page_index,year,default_date,model):
    from openai import OpenAI
    # Limit image size to keep API costs manageable; do not send company/private files automatically.
    ok,encoded=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,87])
    if not ok:raise ValueError('Image encoding failed')
    data_url='data:image/jpeg;base64,'+base64.b64encode(encoded.tobytes()).decode()
    prompt=("Extract LABORATORY MEASUREMENTS from this IID report page into the given JSON schema. "
      "Read columns visually; the numerical Result is NOT the method, reporting limit, MCL, or batch. "
      "Preserve the reported result and units EXACTLY, including ND and <. "
      "Only extract entries belonging to East Highline Canal. If this page is explicitly for another canal, return zero rows. "
      "If the sample station is not visible on this page, return station='not visible'; the parent PDF may contain context. "
      "Do not invent date, result, units, or parameters; omit uncertain rows. "
      "One row per actual analyte/result. Return source_row for audit.")
    response=OpenAI(timeout=90,max_retries=2).chat.completions.create(
      model=model,messages=[{'role':'user','content':[{'type':'text','text':prompt},
          {'type':'image_url','image_url':{'url':data_url,'detail':'high'}}]}],
      response_format={'type':'json_schema','json_schema':SCHEMA},temperature=0)
    payload=json.loads(response.choices[0].message.content)
    station=payload.get('station','')
    if RE_OTHER.search(station) and not RE_EHL.search(station):return [],'Other canal identified: '+station
    sd=payload.get('sample_date') or default_date
    try:sd=datetime.fromisoformat(sd).date().isoformat() if sd else None
    except ValueError:sd=default_date
    out=[]
    for row in payload['rows']:
        nm=pname(row['parameter'])
        if not nm:continue
        result=row['reported_result'].strip()
        if not RE_NUM.fullmatch(result):continue
        unit,factor=normalize_units(nm,row['unit'],result)
        if unit is None:continue
        censored=result.upper().startswith(('ND','N/D','<'))
        try:v=None if censored else float(result.replace(',',''))*factor
        except ValueError:continue
        if v is not None and ((nm=='pH' and not 0<=v<=14) or (nm=='Turbidity' and v<0)):
            continue
        out.append(dict(sample_date=sd,year=int(sd[:4]) if sd else year,parameter=nm,value=v,
         unit=unit,reported_result=result,method=row['method'],source_page=page_index,
         raw_line=row['source_row'][:700],status='needs_review',
         confidence_note='OpenAI vision candidate; human verification required; station on page: '+station[:100]))
    return out,'AI station: '+station


def extract(pdf_bytes,filename,use_ai=False,model='gpt-4.1-mini',max_pages=40):
    pdf=fitz.open(stream=pdf_bytes,filetype='pdf')
    if len(pdf)>max_pages:raise ValueError(f'PDF has {len(pdf)} pages; maximum is {max_pages}')
    ym=RE_YEAR.search(filename);year=int(ym.group(1)) if ym else None
    notes=[];results=[]
    if not year:raise ValueError('Filename must contain a year (e.g. 2013); review sample dates')
    if use_ai and not os.getenv('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY is missing. Set it in Render Environment; never in code.')
    for idx,page in enumerate(pdf,1):
        image=render(page)
        gray=preprocess(image)
        rows=ocr_rows(gray)
        joined='\n'.join(rows)
        text=page.get_text('text',sort=True)
        context=text+'\n'+joined
        if RE_OTHER.search(context[:400]) and not RE_EHL.search(context[:700]):
            notes.append(f'Page {idx}: other station indicated; skipped');continue
        # For multi-station PDFs, local OCR may include other canal rows; use AI with per-page
        # station identification, or quarantine entire ambiguous pages.
        mixed=bool(RE_OTHER.search(context) and RE_EHL.search(context))
        date_match=re.search(r'(?:Sample\s*Date\s*:?\s*)(\d{1,2}/\d{1,2}/\d{2,4})',context,re.I)
        date=None
        if date_match:
            try:date=datetime.strptime(date_match.group(1),'%m/%d/%y' if len(date_match.group(1).split('/')[-1])==2 else '%m/%d/%Y').date().isoformat()
            except ValueError:pass
        local=[] if mixed else local_candidates(rows,idx,year,date,'OCR')
        ai=[]
        if use_ai:
            try:ai,msg=ai_candidates(image,idx,year,date,model);notes.append(f'Page {idx}: {msg}')
            except Exception as e:notes.append(f'Page {idx}: AI failed: {type(e).__name__}: {str(e)[:160]}')
        # Prefer vision candidates for table reading if present; otherwise retain local results
        # when page is unambiguously East Highline. Never pull results from another canal.
        extracted=ai if ai else local
        results.extend(extracted)
        notes.append(f'Page {idx}: CV OCR {len(local)} rows, AI {len(ai)}, selected {len(extracted)}'+
                     ('; multi-canal page quarantined unless AI isolated rows' if mixed else ''))
    seen=set();unique=[]
    for r in results:
        key=(r['source_page'],r['sample_date'],r['parameter'],r['reported_result'],r['unit'])
        if key not in seen:unique.append(r);seen.add(key)
    return unique,'\n'.join(notes)
