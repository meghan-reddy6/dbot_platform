import os
import json
import logging
import hashlib
import urllib.request
import sys
import glob

logger = logging.getLogger("ModelManager")

class ModelManager:
    """
    Centralized Model Manager for AI Pipeline.
    Handles discovery, downloading, checksum validation, file locking, and runtime checks.
    """
    _instance = None
    _models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    _manifest_path = os.path.join(os.path.dirname(__file__), "..", "config", "models_manifest.json")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.manifest = {}
            cls._instance.model_paths = {}
        return cls._instance

    def bootstrap(self):
        """
        Main entrypoint. To be called at application boot.
        """
        logger.info("Bootstrapping Model Manager...")
        os.makedirs(self._models_dir, exist_ok=True)
        
        self._load_manifest()
        self._cleanup_legacy_models()
        self._verify_and_download_models()
        self._runtime_checks()
        logger.info("Model Manager Bootstrap Complete.")

    def _load_manifest(self):
        if not os.path.exists(self._manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self._manifest_path}")
        with open(self._manifest_path, 'r') as f:
            self.manifest = json.load(f)

    def _cleanup_legacy_models(self):
        """
        Targeted cleanup for known obsolete files to prevent arbitrary deletion.
        """
        logger.info("Scanning for obsolete legacy models...")
        # Clean known obsolete patterns
        obsolete_patterns = [
            "haarcascade*.xml",
            "*.zip",
            "1k3d68.onnx",
            "2d106det.onnx",
            "det_500m.onnx",
            "genderage.onnx",
            "*lbp*.xml"
        ]
        
        for pattern in obsolete_patterns:
            search_path = os.path.join(self._models_dir, pattern)
            for file_path in glob.glob(search_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up obsolete file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"Could not delete legacy file {file_path}: {e}")

    def _verify_and_download_models(self):
        for logical_name, meta in self.manifest.items():
            filename = meta.get("filename")
            target_path = os.path.join(self._models_dir, filename)
            url = meta.get("url", "")
            required = meta.get("required", False)
            expected_sha256 = meta.get("sha256", "")
            
            logger.info(f"[MODEL] Checking {logical_name}...")
            
            if os.path.exists(target_path):
                # Validate Checksum if provided
                if expected_sha256 and not self._validate_checksum(target_path, expected_sha256):
                    logger.warning(f"[MODEL] Checksum mismatch for {filename}. Deleting and re-downloading.")
                    os.remove(target_path)
                else:
                    logger.info(f"[MODEL] {logical_name} found and validated.")
                    self.model_paths[logical_name] = target_path
                    continue

            if not url:
                if required:
                    logger.error(f"[MODEL] Required model missing with no URL: {filename}. Please manually place it in models/.")
                    sys.exit(1)
                else:
                    logger.warning(f"[MODEL] Optional model missing: {filename}.")
                continue

            logger.info(f"[MODEL] Downloading {filename}...")
            lock_path = target_path + ".lock"
            
            # Simple file locking logic for multi-process safety
            if os.path.exists(lock_path):
                logger.info(f"[MODEL] Lock file found for {filename}. Waiting for another process to finish download...")
                import time
                while os.path.exists(lock_path):
                    time.sleep(1)
                if os.path.exists(target_path):
                    self.model_paths[logical_name] = target_path
                    continue

            try:
                open(lock_path, 'a').close()  # Touch lock file
                
                # Download with chunking
                urllib.request.urlretrieve(url, target_path)
                
                logger.info(f"[MODEL] Download complete: {filename}.")
                if expected_sha256 and not self._validate_checksum(target_path, expected_sha256):
                    os.remove(target_path)
                    raise Exception(f"Checksum validation failed after download for {filename}.")
                    
                self.model_paths[logical_name] = target_path
                
            except Exception as e:
                logger.error(f"[MODEL] Download failed for {filename}: {e}")
                if os.path.exists(target_path):
                    os.remove(target_path)
                if required:
                    sys.exit(1)
            finally:
                if os.path.exists(lock_path):
                    os.remove(lock_path)

    def _validate_checksum(self, filepath, expected_hash):
        if not expected_hash:
            return True
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_hash

    def _runtime_checks(self):
        """
        Verify that ONNX runtime can actually parse the required ONNX models.
        """
        logger.info("Executing runtime compatibility checks...")
        try:
            import onnxruntime as ort
        except ImportError:
            logger.warning("onnxruntime not installed, skipping ONNX runtime checks.")
            return

        for logical_name, meta in self.manifest.items():
            if meta.get("framework") == "onnx" and logical_name in self.model_paths:
                path = self.model_paths[logical_name]
                try:
                    ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                    logger.info(f"[MODEL] {logical_name} runtime check passed.")
                except Exception as e:
                    logger.error(f"[MODEL] {logical_name} failed ONNX runtime initialization: {e}")
                    if meta.get("required"):
                        sys.exit(1)

    @classmethod
    def get_model_path(cls, logical_name: str) -> str:
        inst = cls()
        if not inst.model_paths:
            # Child process (Windows spawn) safety: reload paths if empty
            if not inst.manifest:
                try:
                    inst._load_manifest()
                except Exception:
                    pass
            for name, meta in inst.manifest.items():
                inst.model_paths[name] = os.path.join(inst._models_dir, meta.get("filename", ""))
        return inst.model_paths.get(logical_name, None)
