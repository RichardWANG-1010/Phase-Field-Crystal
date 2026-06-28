"""
pfc_io.py - Video Recording and IO Utilities for PFC Simulations
视频录制与IO工具模块 - PFC模拟的视频录制和输入输出工具

This module provides video recording functionality for PFC simulations,
including frame capture and video synthesis using ffmpeg.
本模块提供PFC模拟的视频录制功能，包括帧捕获和使用ffmpeg的视频合成。

Designed as a mixin class to be used with PFC solvers.
设计为mixin类，与PFC求解器一起使用。

Features / 功能:
    - In-memory frame caching for efficiency
      内存帧缓存以提高效率
    - ffmpeg-based video synthesis (MP4 format)
      基于ffmpeg的视频合成（MP4格式）
    - Automatic temporary file cleanup
      自动临时文件清理
    - Configurable video parameters (fps, resolution, etc.)
      可配置的视频参数（帧率、分辨率等）

Author: Jinpeng Wang
Department of Material Engineering
"""

# OS module for file and directory operations
# OS模块，用于文件和目录操作
import os

# Temporary file handling
# 临时文件处理
import tempfile

# File operations (copy, move, delete)
# 文件操作（复制、移动、删除）
import shutil

# Subprocess for running external commands (ffmpeg)
# Subprocess用于运行外部命令（ffmpeg）
import subprocess

# Matplotlib for plotting frames
# Matplotlib用于绘制帧
import matplotlib.pyplot as plt


class PFCIO:
    """
    Mixin class providing video recording and IO functionality.
    提供视频录制和IO功能的mixin类。
    
    This class handles frame capture and video synthesis for PFC
    simulation visualizations.
    本类处理PFC模拟可视化的帧捕获和视频合成。
    
    Key features / 关键功能:
        - In-memory frame caching (no disk I/O during simulation)
          内存帧缓存（模拟期间无磁盘IO）
        - ffmpeg-based MP4 video generation
          基于ffmpeg的MP4视频生成
        - Automatic directory and temporary file management
          自动目录和临时文件管理
        - Configurable video quality and frame rate
          可配置的视频质量和帧率
    
    Notes / 说明:
        - Assumes host class provides: phi, steps
          假设宿主类提供：phi, steps
        - capture_frame() can be overridden by subclasses for custom layouts
          capture_frame()可以被子类重写以实现自定义布局
    """
    
    def initialize_io(
        self,
        record_video=True,
        video_output_name="pfc_simulation.mp4",
        video_fps=20,
        video_figsize=(6, 6)
    ):
        """
        Initialize IO infrastructure for video recording.
        初始化视频录制的IO基础设施。
        
        Sets up frame cache and video configuration parameters.
        设置帧缓存和视频配置参数。
        
        Args / 参数:
            record_video (bool, optional): Whether to record video.
                                          Defaults to True.
                                          是否录制视频，默认True。
            video_output_name (str, optional): Output video filename.
                                              Defaults to "pfc_simulation.mp4".
                                              输出视频文件名，默认"pfc_simulation.mp4"。
            video_fps (int, optional): Video frames per second.
                                      Defaults to 20.
                                      视频帧率，默认20。
            video_figsize (tuple, optional): Figure size for each frame.
                                            Defaults to (6, 6).
                                            每帧的图形尺寸，默认(6, 6)。
        
        Initialized attributes / 初始化的属性:
            - record_video: Flag to enable/disable video recording
              启用/禁用视频录制的标志
            - video_output_name: Output video file name
              输出视频文件名
            - video_fps: Video frame rate
              视频帧率
            - video_figsize: Size of each video frame figure
              每帧图形的尺寸
            - frame_cache: List to store in-memory frame images
              用于存储内存帧图像的列表
        """
        # Flag to control video recording
        # 控制视频录制的标志
        self.record_video = record_video
        
        # Output video file name
        # 输出视频文件名
        self.video_output_name = video_output_name
        
        # Video frame rate (frames per second)
        # 视频帧率（每秒帧数）
        self.video_fps = video_fps
        
        # Figure size for each frame (width, height) in inches
        # 每帧的图形尺寸（宽，高），单位英寸
        self.video_figsize = video_figsize
        
        # Initialize empty frame cache list
        # 初始化空的帧缓存列表
        # Frames are stored in memory as PNG bytes for efficiency
        # 帧以PNG字节形式存储在内存中以提高效率
        self.frame_cache = []
    
    def capture_frame(self):
        """
        Capture current density field phi snapshot, store frame in memory
        for later video synthesis.
        捕获当前密度场phi画面，保存内存图像帧用于后期合成视频。
        
        Default implementation: captures density field φ as heatmap.
        Subclasses can override this for custom frame layouts.
        默认实现：将密度场φ捕获为云图。
        子类可以重写此方法以实现自定义帧布局。
        
        Notes / 说明:
            - Only captures if record_video is True
              仅当record_video为True时捕获
            - Frame is stored in memory (frame_cache) not written to disk
              帧存储在内存（frame_cache）中，不写入磁盘
            - Uses coolwarm colormap for density field
              密度场使用coolwarm色图
            - Shows total steps and current frame number in title
              在标题中显示总步数和当前帧号
        """
        # Skip if video recording is disabled
        # 如果视频录制被禁用则跳过
        if not self.record_video:
            return
        
        # ============================================================
        # Create figure and plot density field
        # 创建图形并绘制密度场
        # ============================================================
        
        # Create figure with specified size
        # 创建指定尺寸的图形
        fig, ax = plt.subplots(figsize=self.video_figsize)
        
        # Plot density field as heatmap
        # 将密度场绘制成云图
        # origin="lower" matches grid ij indexing
        # origin="lower"匹配网格ij索引
        # im = image object returned by imshow, used for colorbar
        # im = imshow返回的图像对象，用于颜色条
        im = ax.imshow(self.phi, cmap="coolwarm", origin="lower")
        
        # Add title with simulation info
        # 添加带模拟信息的标题
        # Shows total simulation steps and current sampled frame count
        # 显示总模拟步数和当前采样帧数
        ax.set_title(
            f"Simulation step total: {self.steps}, "
            f"current sampled frame: {len(self.frame_cache)}"
        )
        
        # Add colorbar, bound to the specific axes
        # 添加颜色条，绑定到特定坐标轴
        # Pass im to bind colorbar to the correct data range
        # 传入im以将颜色条绑定到正确的数据范围
        plt.colorbar(im, ax=ax)
        
        # ============================================================
        # Save frame to memory cache
        # 将帧保存到内存缓存
        # ============================================================
        
        # Convert figure to in-memory binary image
        # 将画布转为内存二进制图像，不生成本地临时图片文件
        from io import BytesIO
        
        # Create in-memory bytes buffer
        # 创建内存字节缓冲区
        buf = BytesIO()
        
        # Save figure to buffer as PNG format
        # 将图形保存为PNG格式到缓冲区
        # dpi=120 balances quality and file size
        # dpi=120平衡质量和文件大小
        # bbox_inches="tight" removes excess whitespace
        # bbox_inches="tight"去除多余空白
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        
        # Reset buffer position to beginning
        # 将缓冲区位置重置到开头
        buf.seek(0)
        
        # Append frame bytes to cache list
        # 将帧字节追加到缓存列表
        self.frame_cache.append(buf.getvalue())
        
        # Close figure to free memory
        # 关闭图形以释放内存
        plt.close(fig)
        
    def frames_to_video(self):
        """
        Read all cached frames, call system ffmpeg to synthesize mp4 video,
        automatically clean up temporary files.
        读取内存缓存的所有帧，调用系统ffmpeg合成mp4视频，自动清理临时文件。
        
        Uses ffmpeg for high-quality MP4 video generation.
        使用ffmpeg生成高质量MP4视频。
        
        Notes / 说明:
            - Writes frames to temporary directory for ffmpeg
              将帧写入临时目录供ffmpeg使用
            - Cleans up temporary files after video generation
              视频生成后清理临时文件
            - Clears frame cache after successful generation
              生成成功后清除帧缓存
            - Uses H.264 codec with CRF=18 for high quality
              使用H.264编码器，CRF=18以获得高质量
        """
        # Skip if video recording is disabled or no frames in cache
        # 如果未开启录制或无仿真帧，跳过视频生成
        if not self.record_video or len(self.frame_cache) == 0:
            print("未开启录制/无仿真帧，跳过视频生成")
            print("Video recording disabled or zero frames captured, skip video generation")
            return
        
        # ============================================================
        # Create temporary directory for frame files
        # 创建临时文件夹存放帧图片
        # ============================================================
        
        # Create temporary directory
        # Create temp directory to store frame images
        temp_dir = tempfile.mkdtemp()
        
        try:
            # ============================================================
            # Write cached frames to temporary PNG files
            # 逐帧写入临时png图片
            # ============================================================
            
            # Iterate over all cached frames
            # Write each cached frame to temporary png file
            for idx, frame_bytes in enumerate(self.frame_cache):
                # Construct frame filename with zero-padded index
                # 构造带零填充索引的帧文件名
                # %06d format ensures proper sorting by ffmpeg
                # %06d格式确保ffmpeg正确排序
                frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                
                # Write frame bytes to file
                # 将帧字节写入文件
                with open(frame_path, "wb") as f:
                    f.write(frame_bytes)
            
            # ============================================================
            # Build and execute ffmpeg command
            # ffmpeg合成命令：无损压缩、固定帧率、覆盖旧文件
            # ============================================================
            
            # ffmpeg command: lossless compression, fixed fps, overwrite old file
            # ffmpeg命令：无损压缩、固定帧率、覆盖旧文件
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Auto-overwrite existing video file
                       # 自动覆盖已有视频文件
                "-framerate", str(self.video_fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # Quality parameter, lower = better quality (0~51)
                               # 画质参数，数值越小画质越高(0~51)
                self.video_output_name
            ]
            
            # Execute ffmpeg subprocess
            # 执行ffmpeg命令
            subprocess.run(
                ffmpeg_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Print success message with absolute path
            # 打印成功消息和绝对路径
            print(f"仿真视频已生成：{os.path.abspath(self.video_output_name)}")
            print(f"Simulation video saved at: {os.path.abspath(self.video_output_name)}")
            
        finally:
            # ============================================================
            # Clean up temporary files and cache
            # 仿真结束自动删除临时帧文件夹
            # ============================================================
            
            # Auto delete temp frame folder after video generate
            # 自动删除临时帧文件夹
            # Remove temporary directory and all its contents
            # 移除临时目录及其所有内容
            shutil.rmtree(temp_dir)
            
            # Clear frame cache to free memory
            # 清空内存帧缓存释放内存
            # Clear frame cache to free memory
            self.frame_cache.clear()
