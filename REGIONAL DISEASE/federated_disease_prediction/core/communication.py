"""
Communication Protocols for Federated Learning

This module implements secure communication protocols for transmitting
model updates between clients and the server.
"""

import numpy as np
import json
import base64
import gzip
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import logging


@dataclass
class Message:
    """Represents a message in the federated learning system."""
    message_type: str  # 'model_update', 'global_model', 'heartbeat', etc.
    sender_id: str
    payload: Dict[str, Any]
    timestamp: float
    message_id: str
    signature: Optional[str] = None


class CommunicationProtocol(ABC):
    """Abstract base class for communication protocols."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.compression_enabled = self.config.get('compression_enabled', True)
        self.encryption_enabled = self.config.get('encryption_enabled', True)
    
    @abstractmethod
    def send(self, message: Message, destination: str) -> bool:
        """Send a message to a destination."""
        pass
    
    @abstractmethod
    def receive(self) -> Optional[Message]:
        """Receive a message."""
        pass
    
    def serialize_weights(
        self, 
        weights: Dict[str, np.ndarray],
        compress: bool = True
    ) -> bytes:
        """
        Serialize model weights for transmission.
        
        Args:
            weights: Dictionary of model weights
            compress: Whether to compress the serialized data
            
        Returns:
            Serialized bytes
        """
        # Convert numpy arrays to lists for serialization
        serializable_weights = {
            k: v.tobytes() if isinstance(v, np.ndarray) else v
            for k, v in weights.items()
        }
        
        # Add shape information for reconstruction
        weight_info = {
            'shapes': {k: v.shape for k, v in weights.items()},
            'dtypes': {k: str(v.dtype) for k, v in weights.items()},
            'data': serializable_weights
        }
        
        # Serialize
        serialized = pickle.dumps(weight_info)
        
        # Compress if enabled
        if compress and self.compression_enabled:
            serialized = gzip.compress(serialized)
        
        return serialized
    
    def deserialize_weights(
        self, 
        data: bytes,
        compressed: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Deserialize model weights from bytes.
        
        Args:
            data: Serialized weight data
            compressed: Whether the data is compressed
            
        Returns:
            Dictionary of model weights
        """
        # Decompress if needed
        if compressed and self.compression_enabled:
            data = gzip.decompress(data)
        
        # Deserialize
        weight_info = pickle.loads(data)
        
        # Reconstruct numpy arrays
        weights = {}
        for key, byte_data in weight_info['data'].items():
            shape = weight_info['shapes'][key]
            dtype = np.dtype(weight_info['dtypes'][key])
            weights[key] = np.frombuffer(byte_data, dtype=dtype).reshape(shape)
        
        return weights
    
    def quantize_weights(
        self,
        weights: Dict[str, np.ndarray],
        bits: int = 8
    ) -> Dict[str, np.ndarray]:
        """
        Quantize model weights to reduce communication cost.
        
        Args:
            weights: Dictionary of model weights
            bits: Number of bits for quantization
            
        Returns:
            Quantized weights
        """
        quantized = {}
        
        for key, weight in weights.items():
            # Calculate min and max for scaling
            w_min = np.min(weight)
            w_max = np.max(weight)
            
            if w_max == w_min:
                quantized[key] = weight
                continue
            
            # Scale to [0, 2^bits - 1]
            scale = (2**bits - 1) / (w_max - w_min)
            quantized_weight = np.round((weight - w_min) * scale).astype(np.uint8)
            
            # Store quantization parameters
            quantized[key] = {
                'values': quantized_weight,
                'min': w_min,
                'max': w_max,
                'bits': bits
            }
        
        return quantized
    
    def dequantize_weights(
        self,
        quantized_weights: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Dequantize model weights.
        
        Args:
            quantized_weights: Quantized weight dictionary
            
        Returns:
            Dequantized weights
        """
        weights = {}
        
        for key, q_weight in quantized_weights.items():
            if isinstance(q_weight, dict) and 'values' in q_weight:
                # Dequantize
                values = q_weight['values']
                w_min = q_weight['min']
                w_max = q_weight['max']
                bits = q_weight['bits']
                
                scale = (w_max - w_min) / (2**bits - 1)
                weights[key] = values.astype(np.float32) * scale + w_min
            else:
                weights[key] = q_weight
        
        return weights


class GRPCProtocol(CommunicationProtocol):
    """
    gRPC-based communication protocol.
    
    Efficient binary protocol suitable for high-performance
    federated learning systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.server_address = self.config.get('server_address', 'localhost:8080')
        self.max_message_size = self.config.get('max_message_size', 100 * 1024 * 1024)  # 100MB
        self.channel = None
        self.stub = None
    
    def connect(self) -> None:
        """Establish connection to server."""
        try:
            import grpc
            from . import fl_pb2, fl_pb2_grpc
            
            options = [
                ('grpc.max_send_message_length', self.max_message_size),
                ('grpc.max_receive_message_length', self.max_message_size),
            ]
            
            self.channel = grpc.insecure_channel(
                self.server_address,
                options=options
            )
            self.stub = fl_pb2_grpc.FederatedLearningStub(self.channel)
            self.logger.info(f"Connected to gRPC server at {self.server_address}")
        except ImportError:
            self.logger.error("gRPC not installed. Install with: pip install grpcio")
            raise
    
    def send(self, message: Message, destination: str) -> bool:
        """Send message via gRPC."""
        if self.stub is None:
            self.connect()
        
        try:
            # Serialize message
            serialized_message = self._serialize_message(message)
            
            # Create gRPC request
            request = fl_pb2.MessageRequest(
                message_id=message.message_id,
                sender_id=message.sender_id,
                message_type=message.message_type,
                payload=serialized_message,
                timestamp=message.timestamp
            )
            
            # Send
            response = self.stub.SendMessage(request)
            return response.success
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def receive(self) -> Optional[Message]:
        """Receive message via gRPC (server-side streaming)."""
        # This would be implemented on the server side
        pass
    
    def _serialize_message(self, message: Message) -> bytes:
        """Serialize message to bytes."""
        message_dict = {
            'type': message.message_type,
            'sender': message.sender_id,
            'payload': message.payload,
            'timestamp': message.timestamp,
            'id': message.message_id
        }
        return pickle.dumps(message_dict)


class HTTPSProtocol(CommunicationProtocol):
    """
    HTTPS-based REST communication protocol.
    
    Suitable for web-based deployments and easier to integrate
    with existing infrastructure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.base_url = self.config.get('base_url', 'https://localhost:8443')
        self.api_version = self.config.get('api_version', 'v1')
        self.timeout = self.config.get('timeout', 30)
        self.session = None
    
    def connect(self) -> None:
        """Initialize HTTP session."""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        self.session = requests.Session()
        
        # Configure retries
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        self.logger.info(f"Initialized HTTPS session to {self.base_url}")
    
    def send(self, message: Message, destination: str) -> bool:
        """Send message via HTTPS POST request."""
        if self.session is None:
            self.connect()
        
        url = f"{self.base_url}/api/{self.api_version}/{destination}"
        
        try:
            # Serialize payload
            if 'model_weights' in message.payload:
                message.payload['model_weights'] = base64.b64encode(
                    self.serialize_weights(message.payload['model_weights'])
                ).decode('utf-8')
            
            response = self.session.post(
                url,
                json={
                    'message_id': message.message_id,
                    'sender_id': message.sender_id,
                    'message_type': message.message_type,
                    'payload': message.payload,
                    'timestamp': message.timestamp
                },
                timeout=self.timeout
            )
            
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    def receive(self) -> Optional[Message]:
        """Poll for messages via HTTPS GET request."""
        if self.session is None:
            self.connect()
        
        url = f"{self.base_url}/api/{self.api_version}/poll"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    return Message(
                        message_type=data['message_type'],
                        sender_id=data['sender_id'],
                        payload=data['payload'],
                        timestamp=data['timestamp'],
                        message_id=data['message_id']
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None


class WebSocketProtocol(CommunicationProtocol):
    """
    WebSocket-based bidirectional communication protocol.
    
    Suitable for real-time communication and push notifications.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.server_url = self.config.get('server_url', 'ws://localhost:8765')
        self.websocket = None
        self.message_queue: List[Message] = []
    
    async def connect(self) -> None:
        """Establish WebSocket connection."""
        try:
            import websockets
            self.websocket = await websockets.connect(self.server_url)
            self.logger.info(f"Connected to WebSocket server at {self.server_url}")
        except ImportError:
            self.logger.error("websockets not installed. Install with: pip install websockets")
            raise
    
    async def send(self, message: Message, destination: str) -> bool:
        """Send message via WebSocket."""
        import websockets
        
        if self.websocket is None:
            await self.connect()
        
        try:
            message_data = {
                'type': message.message_type,
                'sender': message.sender_id,
                'destination': destination,
                'payload': message.payload,
                'timestamp': message.timestamp,
                'id': message.message_id
            }
            
            await self.websocket.send(json.dumps(message_data))
            return True
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.error("WebSocket connection closed")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive(self) -> Optional[Message]:
        """Receive message via WebSocket."""
        import websockets
        
        if self.websocket is None:
            await self.connect()
        
        try:
            message_str = await self.websocket.recv()
            data = json.loads(message_str)
            
            return Message(
                message_type=data['type'],
                sender_id=data['sender'],
                payload=data['payload'],
                timestamp=data['timestamp'],
                message_id=data['id']
            )
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.error("WebSocket connection closed")
            return None
        except Exception as e:
            self.logger.error(f"Failed to receive message: {e}")
            return None


def get_communication_protocol(
    protocol_type: str,
    config: Optional[Dict[str, Any]] = None
) -> CommunicationProtocol:
    """
    Factory function to create communication protocol instances.
    
    Args:
        protocol_type: Type of protocol ('grpc', 'https', 'websocket')
        config: Configuration dictionary
        
    Returns:
        CommunicationProtocol instance
    """
    protocols = {
        'grpc': GRPCProtocol,
        'https': HTTPSProtocol,
        'websocket': WebSocketProtocol,
    }
    
    protocol_class = protocols.get(protocol_type.lower())
    if protocol_class is None:
        raise ValueError(f"Unknown protocol type: {protocol_type}")
    
    return protocol_class(config)
