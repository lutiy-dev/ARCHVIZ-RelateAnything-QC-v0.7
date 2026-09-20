import json
import math
import numpy as np

VERSION = "0.7.0"


def _as_regions(value, name):
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be RA_REGIONS dict.")
    for key in ("boxes", "height", "width"):
        if key not in value:
            raise ValueError(f"{name} missing key: {key}")
    boxes = np.asarray(value["boxes"], dtype=np.float32)
    if boxes.size == 0:
        boxes = boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"{name}.boxes must be [N,4] pixel xyxy.")
    h = int(value["height"]); w = int(value["width"])
    if h <= 0 or w <= 0: raise ValueError(f"{name} has invalid image size {w}x{h}.")
    if not np.isfinite(boxes).all(): raise ValueError(f"{name}.boxes contains non-finite values.")
    if len(boxes) and (np.any(boxes[:,2:] <= boxes[:,:2]) or np.any(boxes < 0)
                       or np.any(boxes[:,[0,2]] > w) or np.any(boxes[:,[1,3]] > h)):
        raise ValueError(f"{name}.boxes must have positive area and lie inside image bounds.")
    src = value.get("source_indices", list(range(len(boxes))))
    if len(src) != len(boxes): raise ValueError(f"{name}.source_indices length must match boxes.")
    return boxes, h, w, [int(x) for x in src]


def _norm_geometry(boxes, h, w):
    if len(boxes) == 0:
        z = np.zeros((0,4), dtype=np.float32)
        return z, np.zeros((0,2), dtype=np.float32), np.zeros((0,2), dtype=np.float32)
    n = boxes.copy().astype(np.float32)
    n[:,[0,2]] /= float(w); n[:,[1,3]] /= float(h)
    c = np.stack(((n[:,0]+n[:,2])*0.5, (n[:,1]+n[:,3])*0.5), axis=1)
    s = np.stack((n[:,2]-n[:,0], n[:,3]-n[:,1]), axis=1)
    return n,c,s


def _iou(a,b):
    x1=max(float(a[0]),float(b[0])); y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2])); y2=min(float(a[3]),float(b[3]))
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1); inter=iw*ih
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
    bb=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    u=aa+bb-inter
    return inter/u if u>0 else 0.0


def _match(bn,bc,bs,an,ac,ass,max_center_distance,min_iou):
    cand=[]; diag=math.sqrt(2.0)
    for i in range(len(bn)):
        for j in range(len(an)):
            cd=float(np.linalg.norm(bc[i]-ac[j])/diag); iou=_iou(bn[i],an[j])
            sd=float(np.mean(np.abs(ass[j]-bs[i])/np.maximum(bs[i],1e-6)))
            if cd <= max_center_distance or iou >= min_iou:
                cost=cd+0.35*(1-iou)+0.15*min(sd,2.0)
                cand.append((cost,cd,iou,sd,i,j))
    cand.sort(key=lambda x:x[0])
    ub=set(); ua=set(); matches=[]
    for cost,cd,iou,sd,i,j in cand:
        if i in ub or j in ua: continue
        ub.add(i); ua.add(j)
        matches.append({"before_idx":i,"after_idx":j,"match_cost":float(cost),
                        "center_distance_normdiag":float(cd),"iou":float(iou),
                        "size_delta_ratio":float(sd)})
    return matches,[i for i in range(len(bn)) if i not in ub],[j for j in range(len(an)) if j not in ua]


def _pairwise_structure(matches,bc,ac,row_tol,col_tol,order_tol):
    m={x['before_idx']:x['after_idx'] for x in matches}; ids=sorted(m)
    lr=[]; ud=[]; rows=[]; cols=[]; spacing=[]
    for p in range(len(ids)):
        for q in range(p+1,len(ids)):
            i,k=ids[p],ids[q]; j,l=m[i],m[k]
            bdx=float(bc[k,0]-bc[i,0]); adx=float(ac[l,0]-ac[j,0])
            bdy=float(bc[k,1]-bc[i,1]); ady=float(ac[l,1]-ac[j,1])
            if abs(bdx)>order_tol and abs(adx)>order_tol and np.sign(bdx)!=np.sign(adx): lr.append([i,k])
            if abs(bdy)>order_tol and abs(ady)>order_tol and np.sign(bdy)!=np.sign(ady): ud.append([i,k])
            brow=abs(bdy)<=row_tol; arow=abs(ady)<=row_tol
            if brow!=arow: rows.append([i,k,brow,arow])
            bcol=abs(bdx)<=col_tol; acol=abs(adx)<=col_tol
            if bcol!=acol: cols.append([i,k,bcol,acol])
            spacing.append(abs(float(np.linalg.norm(ac[l]-ac[j]))-float(np.linalg.norm(bc[k]-bc[i]))))
    return {"left_right_order_flips":lr,"above_below_order_flips":ud,
            "same_row_changes":rows,"same_column_changes":cols,
            "pair_spacing_drift_mean_norm":float(np.mean(spacing)) if spacing else 0.0,
            "pair_spacing_drift_max_norm":float(np.max(spacing)) if spacing else 0.0}


def _parse_relations(payload):
    if payload is None: return None
    if isinstance(payload,list):
        if not payload: return None
        payload=payload[0]
    s=str(payload).strip()
    if not s: return None
    d=json.loads(s); r=d.get('relations')
    return r if isinstance(r,list) else None


def _semantic_compare(before_json,after_json,before_src,after_src,matches):
    try: br=_parse_relations(before_json); ar=_parse_relations(after_json)
    except Exception as e: return {"available":False,"reason":f"relations_json parse error: {e}"}
    if br is None or ar is None: return {"available":False,"reason":"relations_json not connected or has no relations array"}
    bp={s:i for i,s in enumerate(before_src)}; ap={s:i for i,s in enumerate(after_src)}
    a2b={m['after_idx']:m['before_idx'] for m in matches}
    def bsig(r):
        s=bp.get(int(r.get('subject_source_idx',-1))); o=bp.get(int(r.get('object_source_idx',-1)))
        return None if s is None or o is None else (s,str(r.get('predicate','')),o)
    def asig(r):
        sa=ap.get(int(r.get('subject_source_idx',-1))); oa=ap.get(int(r.get('object_source_idx',-1)))
        if sa is None or oa is None: return None
        s=a2b.get(sa); o=a2b.get(oa)
        return None if s is None or o is None else (s,str(r.get('predicate','')),o)
    bset={x for x in (bsig(r) for r in br) if x is not None}; aset={x for x in (asig(r) for r in ar) if x is not None}
    return {"available":True,"note":"Advisory semantic comparison only; geometry QC does not depend on RelateAnything scores.",
            "before_relation_count":len(bset),"after_relation_count_mapped":len(aset),
            "removed_relations":[list(x) for x in sorted(bset-aset)],
            "added_relations":[list(x) for x in sorted(aset-bset)]}


class RACompareQCV07:
    RETURN_TYPES=("STRING","STRING")
    RETURN_NAMES=("qc_report","qc_json")
    FUNCTION="compare"
    CATEGORY="RelateAnything/QC v0.7"
    OUTPUT_NODE=True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "before_regions":("RA_REGIONS",),"after_regions":("RA_REGIONS",),
            "max_center_distance":("FLOAT",{"default":0.08,"min":0.005,"max":0.5,"step":0.005}),
            "min_iou":("FLOAT",{"default":0.05,"min":0.0,"max":1.0,"step":0.01}),
            "warn_center_drift_pct":("FLOAT",{"default":1.0,"min":0.0,"max":20.0,"step":0.1}),
            "warn_size_drift_pct":("FLOAT",{"default":8.0,"min":0.0,"max":100.0,"step":0.5}),
            "row_tolerance":("FLOAT",{"default":0.02,"min":0.001,"max":0.2,"step":0.001}),
            "column_tolerance":("FLOAT",{"default":0.02,"min":0.001,"max":0.2,"step":0.001}),
            "order_tolerance":("FLOAT",{"default":0.01,"min":0.001,"max":0.2,"step":0.001})},
            "optional":{"before_relations_json":("STRING",{"forceInput":True}),"after_relations_json":("STRING",{"forceInput":True})}}

    def compare(self,before_regions,after_regions,max_center_distance,min_iou,warn_center_drift_pct,
                warn_size_drift_pct,row_tolerance,column_tolerance,order_tolerance,
                before_relations_json=None,after_relations_json=None):
        bb,bh,bw,bsrc=_as_regions(before_regions,'before_regions'); ab,ah,aw,asrc=_as_regions(after_regions,'after_regions')
        bn,bc,bs=_norm_geometry(bb,bh,bw); an,ac,ass=_norm_geometry(ab,ah,aw)
        matches,missing,added=_match(bn,bc,bs,an,ac,ass,float(max_center_distance),float(min_iou))
        diag=math.sqrt(2.0); cds=[]; sds=[]
        for m in matches:
            i,j=m['before_idx'],m['after_idx']
            cp=float(np.linalg.norm(ac[j]-bc[i])/diag*100.0)
            sp=float(np.mean(np.abs(ass[j]-bs[i])/np.maximum(bs[i],1e-6))*100.0)
            m.update(center_drift_pct_diag=cp,size_drift_pct_mean=sp,before_source_idx=bsrc[i],after_source_idx=asrc[j])
            cds.append(cp); sds.append(sp)
        structure=_pairwise_structure(matches,bc,ac,float(row_tolerance),float(column_tolerance),float(order_tolerance))
        semantic=_semantic_compare(before_relations_json,after_relations_json,bsrc,asrc,matches)
        maxc=max(cds) if cds else 0.0; maxs=max(sds) if sds else 0.0
        hard=bool(missing or added or structure['left_right_order_flips'] or structure['above_below_order_flips'])
        warn=bool(maxc>float(warn_center_drift_pct) or maxs>float(warn_size_drift_pct) or structure['same_row_changes'] or structure['same_column_changes'])
        status='FAIL' if hard else ('WARN' if warn else 'PASS')
        result={"version":VERSION,"status":status,"geometry_truth":"Deterministic bbox comparison. RelateAnything relations are advisory only.",
                "before":{"count":len(bb),"width":bw,"height":bh},"after":{"count":len(ab),"width":aw,"height":ah},
                "matched_count":len(matches),"missing_before_indices":missing,"missing_before_source_indices":[bsrc[i] for i in missing],
                "added_after_indices":added,"added_after_source_indices":[asrc[j] for j in added],
                "center_drift":{"mean_pct_diag":float(np.mean(cds)) if cds else 0.0,"max_pct_diag":maxc,"warn_threshold_pct_diag":float(warn_center_drift_pct)},
                "size_drift":{"mean_pct":float(np.mean(sds)) if sds else 0.0,"max_pct":maxs,"warn_threshold_pct":float(warn_size_drift_pct)},
                "structure":structure,"matches":matches,"semantic_relations":semantic,
                "settings":{"max_center_distance":float(max_center_distance),"min_iou":float(min_iou),"row_tolerance":float(row_tolerance),
                            "column_tolerance":float(column_tolerance),"order_tolerance":float(order_tolerance)}}
        lines=[f"ARCHVIZ QC v0.7 — {status}",f"Regions: BEFORE {len(bb)} → AFTER {len(ab)} | matched {len(matches)}",
               f"Missing: {len(missing)} | Added: {len(added)}",
               f"Center drift: mean {result['center_drift']['mean_pct_diag']:.3f}% | max {maxc:.3f}% of image diagonal",
               f"Size drift: mean {result['size_drift']['mean_pct']:.2f}% | max {maxs:.2f}%",
               f"Order flips: LR {len(structure['left_right_order_flips'])} | UD {len(structure['above_below_order_flips'])}",
               f"Grid changes: rows {len(structure['same_row_changes'])} | columns {len(structure['same_column_changes'])}"]
        if missing: lines.append('Missing BEFORE indices: '+', '.join(map(str,missing)))
        if added: lines.append('Added AFTER indices: '+', '.join(map(str,added)))
        if semantic.get('available'): lines.append(f"RA semantic delta (advisory): -{len(semantic['removed_relations'])} / +{len(semantic['added_relations'])}")
        else: lines.append('RA semantic delta: unavailable ('+semantic.get('reason','unknown')+')')
        lines.append('Geometry decision is based on deterministic bbox math, not RA confidence.')
        text='\n'.join(lines); payload=json.dumps(result,ensure_ascii=False,indent=2)
        print('[ARCHVIZ QC v0.7]\n'+text)
        return {"ui":{"text":[text]},"result":(text,payload)}

NODE_CLASS_MAPPINGS={"RACompareQCV07":RACompareQCV07}
NODE_DISPLAY_NAME_MAPPINGS={"RACompareQCV07":"RA · BEFORE vs AFTER QC · v0.7"}
