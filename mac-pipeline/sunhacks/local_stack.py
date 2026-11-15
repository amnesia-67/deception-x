"""Hash-bound reviewed 89/v2 and 88/v3 digest models with their exact serving policy."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .local_digest import (combine_ensemble_scores,load_digest_schema,digest_contract,
                           ensemble_policy,LEGACY_FEATURE_VERSION)

MANIFEST_SHA256 = "281d0f016737a9b92f3a3623a581300346303224d8adc877e078267c12531516"
MODEL_SHA256 = "ae232e41824d4cdde4742d8a8e6c0db04d92c0b123133a35449675456d95c3ec"
NEW_MANIFEST_SHA256 = "0057caa6a7c6587e48b27900c726276681297875cc093ac204ec755ac7a1c1be"
NEW_MODEL_SHA256 = "f501ea925d0cc2f8d15776b239d97463963f2a83e4a155303420eaed36f563c1"


class LocalDigestStack:
    def __init__(self, manifest_path, model_path, *, expected_manifest_sha256=MANIFEST_SHA256,
                 expected_model_sha256=MODEL_SHA256,feature_version=LEGACY_FEATURE_VERSION,
                 schema_path=None,expected_schema_sha256=None):
        self.manifest_path,self.model_path=Path(manifest_path),Path(model_path)
        self.manifest_sha,self.model_sha=expected_manifest_sha256,expected_model_sha256
        self.feature_version=feature_version
        contract=digest_contract(feature_version)
        self.feature_count=contract['feature_count']
        self.schema_sha=contract['schema_sha256']
        if expected_schema_sha256 is not None and expected_schema_sha256!=self.schema_sha:
            raise ValueError("Schema hash is not the reviewed pin for this feature version")
        self.schema_path=Path(schema_path) if schema_path is not None else Path(__file__).parent/'vendor'/contract['schema_file']
        self.schema=load_digest_schema(feature_version)
        self.model=None;self.manifest=None

    @staticmethod
    def _read(path,expected,max_bytes):
        with path.open("rb") as stream:data=stream.read(max_bytes+1)
        if len(data)>max_bytes or hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError("Local deployment artifact hash/size mismatch: "+path.name)
        return data

    def _validated_manifest(self):
        manifest=json.loads(self._read(self.manifest_path,self.manifest_sha,4*1024*1024))
        policy=ensemble_policy(manifest)
        if policy['feature_version']!=self.feature_version:
            raise ValueError("Ensemble manifest and configured feature version differ")
        schema=json.loads(self._read(self.schema_path,self.schema_sha,65536))
        if schema.get('feature_version')!=self.feature_version or schema.get('feature_names')!=self.schema['feature_names']:
            raise ValueError("Schema version/order differs from the reviewed digest contract")
        if manifest.get('format')=='sunhacks-ensemble-v2':
            if manifest.get('digest_model_sha256')!=self.model_sha or manifest.get('digest_schema_sha256')!=self.schema_sha:
                raise ValueError("Manifest digest model/schema hash binding differs from configured artifacts")
        combine_ensemble_scores(.5,.5,manifest)
        return manifest

    @staticmethod
    def _stack_fields(manifest):
        return {**ensemble_policy(manifest),'cv_metrics':manifest.get('cv_metrics'),
                'caveats':manifest.get('caveats',[])}

    def prepare(self):
        started=time.perf_counter()
        if self.model is None:
            manifest=self._validated_manifest()
            data=self._read(self.model_path,self.model_sha,4*1024*1024)
            import xgboost as xgb
            booster=xgb.Booster(params={"device":"cpu","nthread":2})
            booster.load_model(bytearray(data));booster.set_param({"device":"cpu","nthread":2})
            cfg=json.loads(booster.save_config());names=self.schema['feature_names']
            if (booster.num_features()!=self.feature_count or (booster.feature_names is not None and booster.feature_names!=names)
                    or cfg['learner']['objective']['name']!='binary:logistic'):
                raise ValueError("Unexpected trained digest model objective/schema")
            self.model,self.manifest=booster,manifest
        return {'status':'ready','weights_loaded':True,'kernels_warmed':False,
                'feature_count':self.feature_count,'feature_version':self.feature_version,
                'locality':'mac','elapsed_ms':round((time.perf_counter()-started)*1000,3)}

    def infer(self,row,dasheng_score,*,source_window_id=None):
        import xgboost as xgb
        started=time.perf_counter()
        if self.model is None or self.manifest is None:raise RuntimeError("Local digest/stack has not been prepared")
        if (tuple(row.feature_names)!=tuple(self.schema['feature_names']) or row.values.shape!=(1,self.feature_count)
                or getattr(row,'feature_version',LEGACY_FEATURE_VERSION)!=self.feature_version):
            raise ValueError("Digest schema/order/shape/version changed")
        if np.isinf(row.values).any():raise ValueError("Infinite digest feature")
        value=float(self.model.predict(xgb.DMatrix(row.values,feature_names=list(row.feature_names)))[0])
        if not np.isfinite(value) or not 0<=value<=1:raise ValueError("Invalid local digest score")
        score=combine_ensemble_scores(dasheng_score,value,self.manifest)
        common={'status':'trained_classifier','classifier_only':True,'calibrated':False,
                'source_window_id':source_window_id,'locality':'mac','precision_parity_verified':False,
                'feature_version':self.feature_version}
        return {'tabular':{**common,'model_id':f'teammate/xgb_digest_{self.feature_count}','raw_score':value,
                    'feature_count':self.feature_count,'missing_features':list(row.missing_features),
                    'elapsed_ms':round((time.perf_counter()-started)*1000,3)},
                'stack':{**common,'model_id':'teammate/stacked_ensemble','raw_score':float(score),
                    'used_modalities':['tuned_dasheng','xgb_digest'],'not_used_modalities':['vision'],
                    'weights':dict(zip(self.manifest['branches'],self.manifest['weights'])),
                    'calibration_scope':'parameters_fit_on_courtroom_data;webcam_calibration_unverified',
                    **self._stack_fields(self.manifest)}}

    def close(self):self.model=None;self.manifest=None
