import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import time
import os
import pickle
import zlib
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
from math import exp
from torchvision.models import vgg16

class SSIM(nn.Module):
    """Structural Similarity Index Measure loss"""
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean() if self.size_average else 1 - ssim_map.mean(1).mean(1).mean(1)

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        blocks = []
        blocks.append(vgg16(pretrained=True).features[:4].eval())
        blocks.append(vgg16(pretrained=True).features[4:9].eval())
        blocks.append(vgg16(pretrained=True).features[9:16].eval())
        blocks.append(vgg16(pretrained=True).features[16:23].eval())
        
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
            
        return loss

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.conv(x)

class EnhancedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 16, 3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

class EnhancedSuperResolution(nn.Module):
    def __init__(self, upscale_factor=8):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        self.feature_proj = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1)
        )
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(8)]
        )
        
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU()
        )
        
        self.final = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x, features=None):
        x = self.initial(x)
        
        if features is not None:
            features = F.interpolate(features, size=x.shape[2:], mode='bilinear')
            features = self.feature_proj(features)
            x = x + features
            
        x = self.res_blocks(x)
        x = self.upsample(x)
        
        if x.shape[2] != self.upscale_factor * x.shape[2]:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            
        return torch.sigmoid(self.final(x))

class ExtremeVideoCompressor:
    def __init__(self, scale_factor=0.125, keyframe_interval=30, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.scale_factor = scale_factor
        self.keyframe_interval = keyframe_interval
        self.device = device
        self.frame_count = 0
        self.recent_frames = deque(maxlen=5)
        
        self.feature_extractor = EnhancedFeatureExtractor().to(device)
        self.super_resolution = EnhancedSuperResolution(upscale_factor=int(1/scale_factor)).to(device)
        
        self.jpeg_quality = 15
        self.max_colors = 16
        self.block_size = 16
        self.search_range = 16
        
        self.last_keyframe = None
        self.last_frame = None
        self.last_features = None
        self.compressed_buffer = []
        self.is_trained = False
        
        self.feature_optimizer = optim.Adam(self.feature_extractor.parameters(), lr=0.0005)
        self.sr_optimizer = optim.Adam(self.super_resolution.parameters(), lr=0.0005)
        
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIM().to(device)
        self.vgg_loss = VGGPerceptualLoss().to(device)
        
        self.model_dir = "extreme_model_weights"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self.feature_extractor.apply(weights_init)
        self.super_resolution.apply(weights_init)
    
    def preprocess_frame(self, frame):
        if isinstance(frame, np.ndarray):
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        return frame.to(self.device).unsqueeze(0)
    
    def postprocess_frame(self, tensor):
        tensor = tensor.detach().squeeze(0).cpu()
        tensor = torch.clamp(tensor, 0.0, 1.0)
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img
    
    def augment_frame(self, frame):
        if np.random.rand() > 0.5:
            frame = cv2.flip(frame, 1)
        
        angle = np.random.uniform(-5, 5)
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        frame = cv2.warpAffine(frame, M, (w, h))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame[:,:,1] = frame[:,:,1] * np.random.uniform(0.9, 1.1)
        frame[:,:,2] = frame[:,:,2] * np.random.uniform(0.9, 1.1)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        
        return frame
    
    def reduce_colors(self, image, num_colors=16):
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        if pixels.shape[0] > 10000:
            sample_indices = np.random.choice(pixels.shape[0], 10000, replace=False)
            samples = pixels[sample_indices]
            kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=1).fit(samples)
        else:
            kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=1).fit(pixels)
        
        labels = kmeans.predict(pixels)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        reduced_image = palette[labels].reshape(h, w, c)
        
        return reduced_image, palette
    
    def extract_vector_representation(self, frame, max_points=500):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            for point in approx:
                points.append(point[0])
                if len(points) >= max_points:
                    break
            if len(points) >= max_points:
                break
        
        if len(points) < 50:
            edge_points = np.where(edges > 0)
            if len(edge_points[0]) > 0:
                indices = np.random.choice(len(edge_points[0]), min(50, len(edge_points[0])), replace=False)
                for i in indices:
                    points.append([edge_points[1][i], edge_points[0][i]])
        
        if len(points) == 0:
            h, w = gray.shape
            points = [[w//4, h//4], [3*w//4, h//4], [w//4, 3*h//4], [3*w//4, 3*h//4]]
        
        points = np.array(points[:max_points])
        
        colors = []
        for point in points:
            x, y = point
            if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                if len(frame.shape) == 3:
                    colors.append(frame[y, x])
                else:
                    colors.append([frame[y, x], frame[y, x], frame[y, x]])
        
        colors = np.array(colors, dtype=np.uint8)
        
        return points, colors
    
    def reconstruct_from_vector(self, points, colors, frame_shape):
        h, w = frame_shape[:2]
        reconstructed = np.zeros((h, w, 3), dtype=np.uint8)
        
        if len(points) >= 3:
            try:
                tri = Delaunay(points)
                
                for simplex in tri.simplices:
                    pts = points[simplex]
                    cols = colors[simplex]
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, pts.astype(np.int32), 1)
                    
                    y, x = np.where(mask > 0)
                    avg_color = np.mean(cols, axis=0).astype(np.uint8)
                    reconstructed[y, x] = avg_color
            except Exception as e:
                print(f"Triangulation failed: {e}")
        
        if np.any(reconstructed == 0):
            y_zero, x_zero = np.where(np.all(reconstructed == 0, axis=2))
            
            for i in range(len(y_zero)):
                if i >= len(y_zero):
                    break
                    
                y, x = y_zero[i], x_zero[i]
                dists = np.sqrt((points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2)
                nearest = np.argmin(dists)
                reconstructed[y, x] = colors[nearest]
        
        return reconstructed
    
    def motion_estimation(self, current_frame, reference_frame):
        h, w = current_frame.shape[:2]
        block_size = self.block_size
        search_range = self.search_range
        
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame
            
        if len(reference_frame.shape) == 3:
            reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        else:
            reference_gray = reference_frame
        
        motion_vectors = []
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                bh = min(block_size, h - i)
                bw = min(block_size, w - j)
                
                if bh <= 0 or bw <= 0:
                    continue
                
                block = current_gray[i:i+bh, j:j+bw]
                best_x, best_y = j, i
                best_sad = float('inf')
                
                search_start_x = max(0, j - search_range)
                search_end_x = min(w - bw, j + search_range)
                search_start_y = max(0, i - search_range)
                search_end_y = min(h - bh, i + search_range)
                
                for y in range(search_start_y, search_end_y + 1, 2):
                    for x in range(search_start_x, search_end_x + 1, 2):
                        candidate = reference_gray[y:y+bh, x:x+bw]
                        sad = np.sum(np.abs(block.astype(int) - candidate.astype(int)))
                        
                        if sad < best_sad:
                            best_sad = sad
                            best_x, best_y = x, y
                
                motion_vectors.append((j, i, best_x - j, best_y - i))
        
        return motion_vectors
    
    def apply_motion_compensation(self, reference_frame, motion_vectors, frame_shape):
        h, w = frame_shape[:2]
        block_size = self.block_size
        compensated = np.zeros_like(reference_frame)
        
        for mv in motion_vectors:
            src_x, src_y, dx, dy = mv
            dst_x, dst_y = src_x + dx, src_y + dy
            
            bh = min(block_size, h - src_y, h - dst_y)
            bw = min(block_size, w - src_x, w - dst_x)
            
            if bh <= 0 or bw <= 0:
                continue
            
            compensated[src_y:src_y+bh, src_x:src_x+bw] = reference_frame[dst_y:dst_y+bh, dst_x:dst_x+bw]
        
        return compensated
    
    def compress_keyframe(self, frame):
        frame_tensor = self.preprocess_frame(frame)
        
        with torch.no_grad():
            features = self.feature_extractor(frame_tensor)
            h, w = frame_tensor.shape[2], frame_tensor.shape[3]
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            downsampled = F.interpolate(frame_tensor, size=(new_h, new_w), mode='bilinear')
            self.last_features = features.clone()
        
        tiny_frame = self.postprocess_frame(downsampled)
        
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, jpeg_data = cv2.imencode('.jpg', tiny_frame, encode_param)
        compressed_data = zlib.compress(jpeg_data.tobytes(), level=9)
        
        return {
            'type': 'keyframe',
            'data': compressed_data,
            'shape': (new_h, new_w),
            'original_shape': (h, w)
        }
    
    def compress_pframe(self, frame):
        if self.last_keyframe is None:
            return self.compress_keyframe(frame)
        
        motion_vectors = self.motion_estimation(frame, self.last_frame)
        predicted_frame = self.apply_motion_compensation(self.last_frame, motion_vectors, frame.shape)
        residual = cv2.subtract(frame, predicted_frame)
        reduced_residual, palette = self.reduce_colors(residual, self.max_colors)
        points, colors = self.extract_vector_representation(reduced_residual, max_points=200)
        
        vector_data = {
            'points': points,
            'colors': colors,
            'motion_vectors': motion_vectors,
            'shape': frame.shape[:2]
        }
        
        serialized = pickle.dumps(vector_data)
        compressed_data = zlib.compress(serialized, level=9)
        
        return {
            'type': 'pframe',
            'data': compressed_data,
            'prediction': predicted_frame
        }
    
    def compress_frame(self, frame):
        self.frame_count += 1
        original_frame = frame.copy()
        
        is_keyframe = (self.frame_count % self.keyframe_interval == 1) or (self.last_keyframe is None)
        
        if is_keyframe:
            compressed = self.compress_keyframe(frame)
            self.last_keyframe = frame.copy()
        else:
            compressed = self.compress_pframe(frame)
        
        self.last_frame = frame.copy()
        frame_size = len(compressed['data'])
        compressed['size'] = frame_size
        compressed['original_size'] = frame.shape[0] * frame.shape[1] * (3 if len(frame.shape) > 2 else 1)
        
        self.compressed_buffer.append(compressed)
        print(f"Frame {self.frame_count}: {frame_size / 1024:.2f} KB "
              f"({frame_size / compressed['original_size'] * 100:.2f}% of original)")
        
        return compressed
    
    def decompress_keyframe(self, compressed_data):
        jpeg_data = zlib.decompress(compressed_data['data'])
        nparr = np.frombuffer(jpeg_data, np.uint8)
        tiny_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if tiny_frame is None:
            print("Error: Failed to decode keyframe data")
            return None
        
        frame_tensor = self.preprocess_frame(tiny_frame)
        
        with torch.no_grad():
            h, w = compressed_data['original_shape']
            upscaled = self.super_resolution(frame_tensor, self.last_features)
            
            if upscaled.shape[2] != h or upscaled.shape[3] != w:
                upscaled = F.interpolate(upscaled, size=(h, w), mode='bilinear')
        
        frame = self.postprocess_frame(upscaled)
        self.last_keyframe = frame.copy()
        self.last_frame = frame.copy()
        
        return frame
    
    def decompress_pframe(self, compressed_data):
        if self.last_frame is None:
            print("Error: Cannot decompress P-frame without reference frame")
            return None
        
        serialized = zlib.decompress(compressed_data['data'])
        vector_data = pickle.loads(serialized)
        
        points = vector_data['points']
        colors = vector_data['colors']
        motion_vectors = vector_data['motion_vectors']
        shape = vector_data['shape']
        
        predicted_frame = self.apply_motion_compensation(self.last_frame, motion_vectors, shape)
        residual = self.reconstruct_from_vector(points, colors, shape)
        frame = cv2.add(predicted_frame, residual)
        self.last_frame = frame.copy()
        
        return frame
    
    def decompress_frame(self, compressed_data):
        if compressed_data['type'] == 'keyframe':
            return self.decompress_keyframe(compressed_data)
        else:
            return self.decompress_pframe(compressed_data)
    
    def train_on_frame(self, frame):
        if np.random.rand() > 0.7:
            frame = self.augment_frame(frame)
            
        self.feature_extractor.train()
        self.super_resolution.train()
        
        frame_tensor = self.preprocess_frame(frame)
        features = self.feature_extractor(frame_tensor)
        
        h, w = frame_tensor.shape[2:]
        new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
        downsampled = F.interpolate(frame_tensor, size=(new_h, new_w), mode='bilinear')
        
        self.sr_optimizer.zero_grad()
        self.feature_optimizer.zero_grad()
        
        upscaled = self.super_resolution(downsampled, features)
        
        pixel_loss = self.mse_loss(upscaled, frame_tensor)
        ssim_loss = self.ssim_loss(upscaled, frame_tensor)
        perceptual_loss = self.vgg_loss(upscaled, frame_tensor)
        
        total_loss = pixel_loss + 0.3*ssim_loss + 0.2*perceptual_loss
        
        total_loss.backward()
        self.sr_optimizer.step()
        self.feature_optimizer.step()
        
        return {
            'pixel_loss': pixel_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_on_video(self, input_source, num_frames=300, batch_size=4):
        print(f"Starting training on {input_source}...")
        cap = cv2.VideoCapture(input_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source for training.")
            return False
        
        frames_processed = 0
        losses = []
        
        self.feature_extractor.train()
        self.super_resolution.train()
        
        try:
            while frames_processed < num_frames:
                batch_loss = 0
                batch_frames = []
                
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break
                    
                    batch_frames.append(frame)
                
                for frame in batch_frames:
                    loss_dict = self.train_on_frame(frame)
                    batch_loss += loss_dict['total_loss']
                    frames_processed += 1
                    
                    if frames_processed % 10 == 0:
                        with torch.no_grad():
                            self.feature_extractor.eval()
                            self.super_resolution.eval()
                            
                            frame_tensor = self.preprocess_frame(frame)
                            features = self.feature_extractor(frame_tensor)
                            
                            h, w = frame_tensor.shape[2], frame_tensor.shape[3]
                            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
                            downsampled = F.interpolate(frame_tensor, size=(new_h, new_w), mode='bilinear')
                            
                            upscaled = self.super_resolution(downsampled, features)
                            
                            tiny_frame = self.postprocess_frame(downsampled)
                            reconstructed = self.postprocess_frame(upscaled)
                            
                            tiny_frame_resized = cv2.resize(tiny_frame, (frame.shape[1], frame.shape[0]))
                            comparison = np.hstack([frame, tiny_frame_resized, reconstructed])
                            
                            cv2.putText(comparison, f"Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            cv2.putText(comparison, f"Tiny ({self.scale_factor*100:.0f}%)", (frame.shape[1] + 10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            cv2.putText(comparison, f"Reconstructed", (2*frame.shape[1] + 10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            
                            cv2.putText(comparison, f"Loss: {loss_dict['total_loss']:.4f}", (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            if comparison.shape[1] > 1920:
                                ratio = 1920 / comparison.shape[1]
                                comparison = cv2.resize(comparison, (1920, int(comparison.shape[0] * ratio)))
                            
                            cv2.imshow('Training Progress', comparison)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                break
                            
                            self.feature_extractor.train()
                            self.super_resolution.train()
                
                avg_loss = batch_loss / len(batch_frames) if batch_frames else 0
                losses.append(avg_loss)
                print(f"Frames: {frames_processed}/{num_frames}, Loss: {avg_loss:.4f}")
                
                if frames_processed >= num_frames:
                    break
            
            self.is_trained = True
            self.feature_extractor.eval()
            self.super_resolution.eval()
            self.save_models()
            
            cv2.destroyAllWindows()
            return True   
        except Exception as e:
            print(f"Error during training: {e}")
            cv2.destroyAllWindows()
            return False
        finally:
            cap.release()
    
    def save_models(self):
        torch.save(self.feature_extractor.state_dict(), f"{self.model_dir}/feature_extractor.pth")
        torch.save(self.super_resolution.state_dict(), f"{self.model_dir}/super_resolution.pth")
        print(f"Model weights saved to {self.model_dir} directory")
    
    def load_models(self):
        try:
            self.feature_extractor.load_state_dict(torch.load(f"{self.model_dir}/feature_extractor.pth", map_location=self.device))
            self.super_resolution.load_state_dict(torch.load(f"{self.model_dir}/super_resolution.pth", map_location=self.device))
            self.feature_extractor.eval()
            self.super_resolution.eval()
            self.is_trained = True
            print(f"Model weights loaded from {self.model_dir}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

def process_video(input_path, output_path=None, train=False, num_train_frames=300):
    compressor = ExtremeVideoCompressor()
    
    if train:
        compressor.train_on_video(input_path, num_frames=num_train_frames)
    else:
        if not compressor.load_models():
            print("Warning: Models not trained or missing. Proceeding with untrained models.")
    
    if input_path is None or input_path.lower() == 'webcam':
        print("Using webcam as input...")
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Could not open input video or webcam.")
        return

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            compressed = compressor.compress_frame(frame)
            reconstructed = compressor.decompress_frame(compressed)

            combined = np.hstack((frame, reconstructed))
            cv2.imshow("Original vs Reconstructed", combined)

            if output_path:
                writer.write(reconstructed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

    finally:
        cap.release()
        if output_path:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extreme Video Compression Demo")
    parser.add_argument('--input', type=str, default=None, help="Path to input video file or 'webcam'")
    parser.add_argument('--output', type=str, default=None, help="Path to save decompressed video")
    parser.add_argument('--train', action='store_true', help="Enable training before compression")
    parser.add_argument('--train_frames', type=int, default=20, help="Number of frames for training")
    args = parser.parse_args()

    process_video(args.input, output_path=args.output, train=args.train, num_train_frames=args.train_frames)