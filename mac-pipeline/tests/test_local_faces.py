"""Local geometry/real-detector seams only; no model assets, SDK, sensors or network."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location("staged_local_faces",Path(__file__).resolve().parents[1]/"sunhacks/local_faces.py")
face=importlib.util.module_from_spec(spec);spec.loader.exec_module(face)


class Detector:
    def __init__(self, mode="face"):
        self.mode=mode;self.shapes=[];self.closed=False
    def detect(self,image):
        h,w=image.data.shape[:2];self.shapes.append((h,w))
        detections=[]
        if self.mode=="face" or self.mode=="upscale" and h==48:
            #The higher side*score candidate is selected, not simply the highestscore.
            for x,y,width,height,score in [(0,0,8,8,.99),(4,3,16,12,.8)]:
                detections.append(types.SimpleNamespace(bounding_box=types.SimpleNamespace(
                    origin_x=x,origin_y=y,width=width,height=height),categories=[types.SimpleNamespace(score=score)]))
        return types.SimpleNamespace(detections=detections)
    def close(self):self.closed=True


def adapter(tmp_path,monkeypatch,mode="face"):
    path=tmp_path/'fixture.tflite';path.write_bytes(b'not a model: fake detector fixture only')
    detector=Detector(mode)
    mp=types.SimpleNamespace(Image=lambda **kw:types.SimpleNamespace(**kw),ImageFormat=types.SimpleNamespace(SRGB='rgb'))
    monkeypatch.setattr(face,'_create_detector',lambda path:(mp,detector))
    a=face.LocalFacePreprocessor(path,hashlib.sha256(path.read_bytes()).hexdigest())
    return a,detector


def inputs():
    f=np.zeros((24,32,3),np.uint8);f[:,:,0]=np.arange(32,dtype=np.uint8)[None,:]*7
    return [f]*128,np.arange(128,dtype=np.float64)/16


def source_geometry():
    source=(Path(__file__).parent/'fixtures/reviewed_face_geometry.py').read_text()
    tree=ast.parse(source)
    selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('smooth_boxes','crop_frames')],type_ignores=[])
    env={'np':np,'FACE_SCALE':1.6,'OUT_RES':384}
    exec(compile(selected,'reviewed_geometry_fixture','exec'),env)
    return env


def test_lazy_preparation_and_close_release_detector(tmp_path,monkeypatch):
    a,det=adapter(tmp_path,monkeypatch)
    assert a._detector is None
    assert a.prepare()['cache_hit'] is False
    assert not det.shapes
    assert a.prepare()['cache_hit'] is True
    assert a.close()=={'status':'closed'} and det.closed
    a.close()
    with pytest.raises(face.LocalFaceUnavailable,match='closed'):a.prepare()


def test_largest_side_times_score_and_shapes(tmp_path,monkeypatch):
    a,det=adapter(tmp_path,monkeypatch)
    frames,ts=inputs();crops,meta=a(frames,ts)
    assert crops.shape==(128,384,384,3) and crops.dtype==np.uint8
    assert len(det.shapes)==128 and meta['face_detect_rate']==1
    assert meta['centre_crop_fallback'] is False
    assert np.allclose(meta['boxes_cxcys'],[12,9,16])
    assert np.array_equal(meta['timestamps'],ts)


def test_upscale_coordinates_return_to_original_pixels(tmp_path,monkeypatch):
    a,det=adapter(tmp_path,monkeypatch,'upscale')
    crops,meta=a(*inputs())
    assert len(det.shapes)==256 and det.shapes[:2]==[(24,32),(48,64)]
    assert np.allclose(meta['boxes_cxcys'],[6,4.5,8])
    assert meta['face_detect_rate']==1 and not meta['centre_crop_fallback']


def test_actual_no_detection_result_has_explicit_source_center_fallback(tmp_path,monkeypatch):
    a,det=adapter(tmp_path,monkeypatch,'none')
    frames,ts=inputs();crops,meta=a(frames,ts)
    reference=source_geometry()
    boxes,rate,fallback=reference['smooth_boxes'](np.full((128,3),np.nan,np.float32),32,24)
    expected=reference['crop_frames'](np.stack(frames),boxes)
    assert np.array_equal(crops,expected)
    assert np.array_equal(np.asarray(meta['boxes_cxcys'],np.float32),boxes)
    assert rate==meta['face_detect_rate']==0 and fallback==meta['centre_crop_fallback'] is True


def test_smoothing_interpolation_and_padded_crop_match_reviewed_source():
    reference=source_geometry()
    raw=np.array([[np.nan,np.nan,np.nan],[1,2,30],[100,10,50],[np.nan,np.nan,np.nan],[-3,5,40]],np.float32)
    actual=face.smooth_boxes(raw,32,24);expected=reference['smooth_boxes'](raw,32,24)
    assert np.array_equal(actual[0],expected[0]) and actual[1:]==expected[1:]
    frames=np.stack(inputs()[0][:5]);boxes=np.array([[0,0,40]]*5,np.float32)
    assert np.array_equal(face.crop_frames(frames,boxes),reference['crop_frames'](frames,boxes))


def test_fit_preserves_aspect_exact_integer_sizing():
    f=np.zeros((1001,2001,3),np.uint8)
    assert face._fit(f).shape==(int(1001*960/2001),960,3)


@pytest.mark.parametrize('change',['count','clock_nan','clock_reverse','clock_constant','dtype','vary_shape'])
def test_invalid_inputs_fail_without_loading_detector(tmp_path,monkeypatch,change):
    a,det=adapter(tmp_path,monkeypatch)
    frames,ts=inputs()
    if change=='count':frames=frames[:-1]
    if change=='clock_nan':ts[3]=np.nan
    if change=='clock_reverse':ts[3]=-1
    if change=='clock_constant':ts[:]=0
    if change=='dtype':frames[0]=frames[0].astype(float)
    if change=='vary_shape':frames[0]=np.zeros((2,3,3),np.uint8)
    with pytest.raises(face.LocalFaceUnavailable):a(frames,ts)
    assert a._detector is None and not det.shapes


def test_missing_asset_and_wrong_hash_do_not_instantiate_sdk(tmp_path,monkeypatch):
    monkeypatch.setattr(face,'_create_detector',lambda path:pytest.fail('Loaded SDK for absent/mismatched asset'))
    with pytest.raises(face.LocalFaceUnavailable,match='missing'):face.LocalFacePreprocessor(tmp_path/'absent').prepare()
    p=tmp_path/'model';p.write_bytes(b'fixture')
    with pytest.raises(face.LocalFaceUnavailable,match='SHA256'):face.LocalFacePreprocessor(p,'0'*64).prepare()
