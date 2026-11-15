"""Small bridge from a captured answer to its bounded contradiction task."""
import base64
import io


class ClaimPipelineMixin:
    def _begin_claim_check(self,clip,window_id,stop_event,visual_authorized):
        with self.lock:self._claim_input_preview=None
        config=self.config.get('claim_check',{})
        if not config.get('enabled',False):return None
        from .claim_runtime import ClaimRuntime
        with self.lock:
            if self.claim_runtime is None:self.claim_runtime=ClaimRuntime(config)
            runtime=self.claim_runtime
            session=self.session_id
        images=()
        if visual_authorized and not stop_event.is_set():
            from PIL import Image
            encoded=[]
            for index in (len(clip)//3,2*len(clip)//3):
                output=io.BytesIO();image=Image.fromarray(clip[index]);image.thumbnail((512,512))
                image.save(output,format='JPEG',quality=90,subsampling=0,optimize=True)
                encoded.append(base64.b64encode(output.getvalue()).decode('ascii'))
            images=tuple(encoded)
            with self.lock:
                if not stop_event.is_set() and self.session_id==session:
                    self._claim_input_preview={"session_id":session,"source_window_id":window_id,
                        "images":images,"selected_frame_indices":[len(clip)//3,2*len(clip)//3]}
        return runtime.begin(images=images,visual_authorized=visual_authorized,
                             source_window_id=window_id,stop_event=stop_event)

    def claim_input_preview(self):
        """Explicit loopback-only inspection; never included in polled public state.

        Retain just the two encoded observations for the latest answer in memory.
        The fingerprint proves correspondence with our request, not remote vision
        correctness or independent attestation by the provider.
        """
        from .claim_check import input_fingerprint
        unavailable={"status":"unavailable","reason":"No matching completed Drex input preview. Run a new answer with the photo check enabled."}
        with self.lock:
            preview=getattr(self,"_claim_input_preview",None)
            latest=self.latest or {};claim=latest.get("claim_check") or {}
            window_id=(latest.get("window") or {}).get("id")
            text=(latest.get("transcript") or {}).get("text")
            if (not preview or self.answer.get("status")!="complete" or self.stop_event.is_set()
                    or preview["session_id"]!=self.session_id
                    or type(window_id) is not int or window_id!=preview["source_window_id"]
                    or type(claim.get("source_window_id")) is not int or claim["source_window_id"]!=window_id
                    or latest.get("visual_claim_check") is not True or latest.get("claim_check_enabled") is not True
                    or claim.get("call_made") is not True or claim.get("transmission_status")!="response_received"
                    or claim.get("sent_keyframes")!=2 or claim.get("sent_transcript") is not True
                    or claim.get("status") not in ("contradiction","no_contradiction","unverifiable")
                    or not isinstance(text,str) or not text
                    or input_fingerprint(text,preview["images"])!=claim.get("input_sha256")):
                return unavailable
            return {"status":"available","session_id":self.session_id,"source_window_id":window_id,
                "transcript":text,"images":list(preview["images"]),
                "selected_frame_indices":list(preview["selected_frame_indices"]),
                "input_sha256":claim["input_sha256"],
                "claim_check":{key:claim.get(key) for key in ("status","decision_choice","model","decision_probabilities","elapsed_ms")},
                "decision_effect":"contradiction_only"}
