import numpy as np
import pandas as pd
import cv2
import json
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import xarray as xr
import netCDF4 as nc
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TCCFeatures:
    """Data class to store TCC features"""
    cluster_id: str
    timestamp: datetime
    lat_center: float
    lon_center: float
    lat_coldest: float
    lon_coldest: float
    area_km2: float
    pixel_count: int
    min_brightness_temp: float
    mean_brightness_temp: float
    max_brightness_temp: float
    std_brightness_temp: float
    cloud_top_height: float
    convective_intensity: float
    eccentricity: float
    compactness: float
    movement_speed: float
    movement_direction: float

class INSATDataProcessor:
    """Class to handle INSAT-3D IRBRT data processing"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.pixel_size_km = 4.0  # INSAT-3D IR pixel resolution
        
    def load_insat_data(self, file_path: str) -> xr.Dataset:
        """Load INSAT-3D IRBRT data from NetCDF file"""
        try:
            ds = xr.open_dataset(file_path)
            return ds
        except Exception as e:
            logger.error(f"Error loading INSAT data: {e}")
            return None
    
    def preprocess_ir_data(self, ir_data: np.ndarray) -> np.ndarray:
        """Preprocess IR brightness temperature data"""
        # Convert to Celsius if in Kelvin
        if np.mean(ir_data) > 200:
            ir_data = ir_data - 273.15
        
        # Apply quality control
        ir_data = np.where(ir_data < -100, np.nan, ir_data)
        ir_data = np.where(ir_data > 60, np.nan, ir_data)
        
        # Fill missing values with interpolation
        mask = ~np.isnan(ir_data)
        if np.any(mask):
            ir_data = self._interpolate_missing(ir_data, mask)
        
        return ir_data
    
    def _interpolate_missing(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Interpolate missing values"""
        from scipy.interpolate import griddata
        
        h, w = data.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        valid_points = np.column_stack((x[mask], y[mask]))
        valid_values = data[mask]
        
        missing_points = np.column_stack((x[~mask], y[~mask]))
        
        if len(missing_points) > 0:
            interpolated = griddata(valid_points, valid_values, missing_points, method='linear')
            data[~mask] = interpolated
        
        return data

class TCCDetector:
    """Deep Learning model for TCC detection"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (256, 256, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()
        self.threshold_temp = -40  # Temperature threshold for cold clouds
        
    def _build_model(self) -> tf.keras.Model:
        """Build U-Net model for TCC segmentation"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        
        # Bottleneck
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
        c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
        
        # Decoder
        u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
        u5 = layers.concatenate([u5, c3])
        c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
        c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
        
        u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c2])
        c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
        c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
        
        u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c1])
        c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
        c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
        
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c7)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def detect_cold_clouds(self, ir_data: np.ndarray) -> np.ndarray:
        """Detect cold clouds using temperature threshold"""
        cold_cloud_mask = ir_data < self.threshold_temp
        return cold_cloud_mask.astype(np.uint8)
    
    def segment_clusters(self, ir_data: np.ndarray, cold_cloud_mask: np.ndarray) -> np.ndarray:
        """Segment individual cloud clusters"""
        # Apply morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(cold_cloud_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(cleaned_mask)
        
        # Filter clusters by size (minimum area threshold)
        min_area = 100  # pixels
        filtered_labels = np.zeros_like(labels)
        cluster_id = 1
        
        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) >= min_area:
                filtered_labels[mask] = cluster_id
                cluster_id += 1
        
        return filtered_labels

class FeatureExtractor:
    """Extract features from detected TCCs"""
    
    def __init__(self, pixel_size_km: float = 4.0):
        self.pixel_size_km = pixel_size_km
    
    def extract_features(self, ir_data: np.ndarray, cluster_labels: np.ndarray, 
                        lat_grid: np.ndarray, lon_grid: np.ndarray, 
                        timestamp: datetime) -> List[TCCFeatures]:
        """Extract features for each TCC cluster"""
        features_list = []
        
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == 0:  # Skip background
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_pixels = np.where(cluster_mask)
            
            # Basic geometric features
            pixel_count = np.sum(cluster_mask)
            area_km2 = pixel_count * (self.pixel_size_km ** 2)
            
            # Temperature statistics
            cluster_temps = ir_data[cluster_mask]
            min_temp = np.min(cluster_temps)
            mean_temp = np.mean(cluster_temps)
            max_temp = np.max(cluster_temps)
            std_temp = np.std(cluster_temps)
            
            # Location features
            cluster_lats = lat_grid[cluster_mask]
            cluster_lons = lon_grid[cluster_mask]
            
            lat_center = np.mean(cluster_lats)
            lon_center = np.mean(cluster_lons)
            
            # Find coldest point
            coldest_idx = np.argmin(cluster_temps)
            lat_coldest = cluster_lats[coldest_idx]
            lon_coldest = cluster_lons[coldest_idx]
            
            # Cloud top height estimation (simple approximation)
            cloud_top_height = self._estimate_cloud_height(min_temp)
            
            # Convective intensity
            convective_intensity = max(0, 40 - min_temp)  # Higher for colder clouds
            
            # Shape features
            eccentricity = self._calculate_eccentricity(cluster_mask)
            compactness = self._calculate_compactness(cluster_mask)
            
            # Create feature object
            features = TCCFeatures(
                cluster_id=f"TCC_{timestamp.strftime('%Y%m%d_%H%M')}_{cluster_id:03d}",
                timestamp=timestamp,
                lat_center=lat_center,
                lon_center=lon_center,
                lat_coldest=lat_coldest,
                lon_coldest=lon_coldest,
                area_km2=area_km2,
                pixel_count=pixel_count,
                min_brightness_temp=min_temp,
                mean_brightness_temp=mean_temp,
                max_brightness_temp=max_temp,
                std_brightness_temp=std_temp,
                cloud_top_height=cloud_top_height,
                convective_intensity=convective_intensity,
                eccentricity=eccentricity,
                compactness=compactness,
                movement_speed=0.0,  # Will be calculated by tracker
                movement_direction=0.0  # Will be calculated by tracker
            )
            
            features_list.append(features)
        
        return features_list
    
    def _estimate_cloud_height(self, min_temp: float) -> float:
        """Estimate cloud top height from brightness temperature"""
        # Simple approximation: assume standard atmosphere
        # Temperature decreases ~6.5°C per km
        surface_temp = 30  # Typical tropical surface temperature
        lapse_rate = 6.5  # °C/km
        
        height = max(0, (surface_temp - min_temp) / lapse_rate)
        return height
    
    def _calculate_eccentricity(self, mask: np.ndarray) -> float:
        """Calculate eccentricity of cluster shape"""
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                ellipse = cv2.fitEllipse(contours[0])
                a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
                if a > 0 and b > 0:
                    eccentricity = np.sqrt(1 - (min(a, b) / max(a, b)) ** 2)
                    return eccentricity
        except:
            pass
        return 0.0
    
    def _calculate_compactness(self, mask: np.ndarray) -> float:
        """Calculate compactness of cluster shape"""
        area = np.sum(mask)
        if area > 0:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter ** 2)
                    return compactness
        return 0.0

class TCCTracker:
    """Track TCCs across time using spatial-temporal correlation"""
    
    def __init__(self, max_distance_km: float = 200, max_time_hours: float = 3):
        self.max_distance_km = max_distance_km
        self.max_time_hours = max_time_hours
        self.active_tracks = {}
        self.track_history = defaultdict(list)
        
    def track_clusters(self, current_features: List[TCCFeatures], 
                      previous_features: List[TCCFeatures] = None) -> Dict[str, List[TCCFeatures]]:
        """Track clusters between time steps"""
        if previous_features is None:
            # Initialize tracks for first time step
            for feature in current_features:
                self.active_tracks[feature.cluster_id] = feature
                self.track_history[feature.cluster_id].append(feature)
            return self.track_history
        
        # Match current features with previous tracks
        matched_pairs = self._match_clusters(current_features, previous_features)
        
        # Update existing tracks and create new ones
        updated_tracks = {}
        for current_feature in current_features:
            matched_previous = None
            for current_id, previous_id in matched_pairs:
                if current_id == current_feature.cluster_id:
                    matched_previous = previous_id
                    break
            
            if matched_previous:
                # Update existing track
                track_id = self._get_track_id(matched_previous)
                updated_tracks[track_id] = current_feature
                
                # Calculate movement
                previous_feature = self.active_tracks[matched_previous]
                current_feature.movement_speed = self._calculate_movement_speed(
                    previous_feature, current_feature
                )
                current_feature.movement_direction = self._calculate_movement_direction(
                    previous_feature, current_feature
                )
                
                self.track_history[track_id].append(current_feature)
            else:
                # Create new track
                new_track_id = current_feature.cluster_id
                updated_tracks[new_track_id] = current_feature
                self.track_history[new_track_id].append(current_feature)
        
        self.active_tracks = updated_tracks
        return self.track_history
    
    def _match_clusters(self, current_features: List[TCCFeatures], 
                       previous_features: List[TCCFeatures]) -> List[Tuple[str, str]]:
        """Match clusters between time steps based on spatial proximity"""
        matches = []
        
        for current_feature in current_features:
            best_match = None
            min_distance = float('inf')
            
            for previous_feature in previous_features:
                distance = self._calculate_distance(
                    current_feature.lat_center, current_feature.lon_center,
                    previous_feature.lat_center, previous_feature.lon_center
                )
                
                time_diff = abs((current_feature.timestamp - previous_feature.timestamp).total_seconds() / 3600)
                
                if distance < self.max_distance_km and time_diff < self.max_time_hours:
                    if distance < min_distance:
                        min_distance = distance
                        best_match = previous_feature.cluster_id
            
            if best_match:
                matches.append((current_feature.cluster_id, best_match))
        
        return matches
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km"""
        from math import radians, cos, sin, asin, sqrt
        
        # Haversine formula
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth's radius in km
        return c * r
    
    def _calculate_movement_speed(self, previous: TCCFeatures, current: TCCFeatures) -> float:
        """Calculate movement speed in km/h"""
        distance = self._calculate_distance(
            previous.lat_center, previous.lon_center,
            current.lat_center, current.lon_center
        )
        time_diff = (current.timestamp - previous.timestamp).total_seconds() / 3600
        return distance / time_diff if time_diff > 0 else 0.0
    
    def _calculate_movement_direction(self, previous: TCCFeatures, current: TCCFeatures) -> float:
        """Calculate movement direction in degrees"""
        from math import atan2, degrees, radians, cos, sin
        
        lat1, lon1 = radians(previous.lat_center), radians(previous.lon_center)
        lat2, lon2 = radians(current.lat_center), radians(current.lon_center)
        
        dlon = lon2 - lon1
        y = sin(dlon) * cos(lat2)
        x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
        
        bearing = atan2(y, x)
        return (degrees(bearing) + 360) % 360
    
    def _get_track_id(self, cluster_id: str) -> str:
        """Get track ID for a cluster"""
        for track_id, features in self.track_history.items():
            if any(f.cluster_id == cluster_id for f in features):
                return track_id
        return cluster_id

class ITCCDatabase:
    """Database manager for ITCC dataset"""
    
    def __init__(self, db_path: str = "itcc_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tcc_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id TEXT NOT NULL,
                track_id TEXT,
                timestamp DATETIME NOT NULL,
                lat_center REAL,
                lon_center REAL,
                lat_coldest REAL,
                lon_coldest REAL,
                area_km2 REAL,
                pixel_count INTEGER,
                min_brightness_temp REAL,
                mean_brightness_temp REAL,
                max_brightness_temp REAL,
                std_brightness_temp REAL,
                cloud_top_height REAL,
                convective_intensity REAL,
                eccentricity REAL,
                compactness REAL,
                movement_speed REAL,
                movement_direction REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON tcc_features(timestamp);
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_cluster_id ON tcc_features(cluster_id);
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_features(self, features_list: List[TCCFeatures], track_id: str = None):
        """Insert TCC features into database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for feature in features_list:
            cursor.execute('''
                INSERT INTO tcc_features (
                    cluster_id, track_id, timestamp, lat_center, lon_center,
                    lat_coldest, lon_coldest, area_km2, pixel_count,
                    min_brightness_temp, mean_brightness_temp, max_brightness_temp,
                    std_brightness_temp, cloud_top_height, convective_intensity,
                    eccentricity, compactness, movement_speed, movement_direction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feature.cluster_id, track_id, feature.timestamp,
                feature.lat_center, feature.lon_center,
                feature.lat_coldest, feature.lon_coldest,
                feature.area_km2, feature.pixel_count,
                feature.min_brightness_temp, feature.mean_brightness_temp,
                feature.max_brightness_temp, feature.std_brightness_temp,
                feature.cloud_top_height, feature.convective_intensity,
                feature.eccentricity, feature.compactness,
                feature.movement_speed, feature.movement_direction
            ))
        
        conn.commit()
        conn.close()
    
    def export_to_csv(self, output_path: str, start_date: datetime = None, end_date: datetime = None):
        """Export ITCC dataset to CSV"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM tcc_features"
        params = []
        
        if start_date or end_date:
            query += " WHERE "
            conditions = []
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date)
            query += " AND ".join(conditions)
        
        query += " ORDER BY timestamp"
        
        df = pd.read_sql_query(query, conn, params=params)
        df.to_csv(output_path, index=False)
        conn.close()
        
        logger.info(f"Exported {len(df)} records to {output_path}")

class ITCCSystem:
    """Main ITCC detection and tracking system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.data_processor = INSATDataProcessor(self.config['data_path'])
        self.detector = TCCDetector()
        self.feature_extractor = FeatureExtractor()
        self.tracker = TCCTracker()
        self.database = ITCCDatabase(self.config['database_path'])
        
        logger.info("ITCC System initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'data_path': './insat_data/',
            'database_path': 'itcc_database.db',
            'output_path': './output/',
            'min_cluster_area': 100,
            'temperature_threshold': -40,
            'tracking_distance': 200,
            'tracking_time': 3
        }
    
    def process_file(self, file_path: str, timestamp: datetime) -> List[TCCFeatures]:
        """Process a single INSAT file"""
        logger.info(f"Processing file: {file_path}")
        
        # Load and preprocess data
        dataset = self.data_processor.load_insat_data(file_path)
        if dataset is None:
            return []
        
        # Extract IR data and coordinates
        ir_data = dataset['IRBRT'].values  # Adjust variable name as needed
        lat_grid = dataset['lat'].values
        lon_grid = dataset['lon'].values
        
        # Preprocess IR data
        ir_data = self.data_processor.preprocess_ir_data(ir_data)
        
        # Detect cold clouds
        cold_cloud_mask = self.detector.detect_cold_clouds(ir_data)
        
        # Segment clusters
        cluster_labels = self.detector.segment_clusters(ir_data, cold_cloud_mask)
        
        # Extract features
        features = self.feature_extractor.extract_features(
            ir_data, cluster_labels, lat_grid, lon_grid, timestamp
        )
        
        logger.info(f"Detected {len(features)} TCCs")
        return features
    
    def run_batch_processing(self, file_list: List[Tuple[str, datetime]]):
        """Process multiple files in batch"""
        logger.info(f"Starting batch processing of {len(file_list)} files")
        
        previous_features = None
        
        for i, (file_path, timestamp) in enumerate(file_list):
            try:
                # Process current file
                current_features = self.process_file(file_path, timestamp)
                
                if current_features:
                    # Track clusters
                    tracks = self.tracker.track_clusters(current_features, previous_features)
                    
                    # Store in database
                    for track_id, track_features in tracks.items():
                        if track_features:  # Only store non-empty tracks
                            self.database.insert_features([track_features[-1]], track_id)
                    
                    previous_features = current_features
                
                logger.info(f"Processed {i+1}/{len(file_list)} files")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        logger.info("Batch processing completed")
    
    def export_dataset(self, output_path: str, format: str = 'csv'):
        """Export ITCC dataset"""
        if format.lower() == 'csv':
            self.database.export_to_csv(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def generate_report(self) -> Dict:
        """Generate processing report"""
        conn = sqlite3.connect(self.database.db_path)
        
        # Basic statistics
        total_clusters = pd.read_sql_query("SELECT COUNT(*) as count FROM tcc_features", conn).iloc[0]['count']
        unique_tracks = pd.read_sql_query("SELECT COUNT(DISTINCT track_id) as count FROM tcc_features", conn).iloc[0]['count']
        
        # Time range
        time_range = pd.read_sql_query(
            "SELECT MIN(timestamp) as start_time, MAX(timestamp) as end_time FROM tcc_features", 
            conn
        ).iloc[0]
        
        # Intensity statistics
        intensity_stats = pd.read_sql_query(
            "SELECT AVG(convective_intensity) as avg_intensity, MAX(convective_intensity) as max_intensity FROM tcc_features",
            conn
        ).iloc[0]
        
        conn.close()
        
        report = {
            'total_clusters_detected': total_clusters,
            'unique_tracks': unique_tracks,
            'time_range': {
                'start': time_range['start_time'],
                'end': time_range['end_time']
            },
            'intensity_statistics': {
                'average_intensity': intensity_stats['avg_intensity'],
                'maximum_intensity': intensity_stats['max_intensity']
            }
        }
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    system = ITCCSystem()
    
    # Example: Process a list of files
    file_list = [
        ("insat_data/file1.nc", datetime(2024, 1, 1, 0, 0)),
        ("insat_data/file2.nc", datetime(2024, 1, 1, 0, 30)),
        ("insat_data/file3.nc", datetime(2024, 1, 1, 1, 0)),
    ]
    
    # Run batch processing
    system.run_batch_processing(file_list)
    
    # Export dataset
    system.export_dataset("itcc_dataset.csv")
    
    # Generate report
    report = system.generate_report()
    print("Processing Report:")
    print(json.dumps(report, indent=2, default=str))