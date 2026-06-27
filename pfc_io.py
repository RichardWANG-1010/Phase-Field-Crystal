import os
import tempfile
import shutil
import subprocess
import matplotlib.pyplot as plt


class PFCIO:
    
    def initialize_io(
        self,
        record_video=True,
        video_output_name="pfc_simulation.mp4",
        video_fps=20,
        video_figsize=(6,6)
    ):

        self.record_video = record_video

        self.video_output_name = video_output_name

        self.video_fps = video_fps

        self.video_figsize = video_figsize

        self.frame_cache = []

    def capture_frame(self):
        """
        捕获当前密度场phi画面，保存内存图像帧用于后期合成视频
        Capture current density field phi snapshot, store frame in memory for video synthesis
        """
        if not self.record_video:
                return
        # 创建临时画布，不弹出可视化窗口
        fig, ax = plt.subplots(figsize=self.video_figsize)
        # 接收imshow返回的图像对象im
        im = ax.imshow(self.phi, cmap="coolwarm", origin="lower")
        ax.set_title(f"Simulation step total: {self.steps}, current sampled frame: {len(self.frame_cache)}")
        # 传入im，指定绑定的坐标轴
        plt.colorbar(im, ax=ax)
        # 将画布转为内存二进制图像，不生成本地临时图片文件
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        self.frame_cache.append(buf.getvalue())
        plt.close(fig)
        
    def frames_to_video(self):
        """
        读取内存缓存的所有帧，调用系统ffmpeg合成mp4视频，自动清理临时文件
        Read all cached frames, call system ffmpeg to synthesize mp4, auto clean temp files
        """
        if not self.record_video or len(self.frame_cache) == 0:
            print("未开启录制/无仿真帧，跳过视频生成")
            print("Video recording disabled or zero frames captured, skip video generation")
            return
        
        # 创建临时文件夹存放帧图片
        # Create temp directory to store frame images
        temp_dir = tempfile.mkdtemp()
        try:
            # 逐帧写入临时png图片
            # Write each cached frame to temporary png file
            for idx, frame_bytes in enumerate(self.frame_cache):
                frame_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                with open(frame_path, "wb") as f:
                    f.write(frame_bytes)
            
            # ffmpeg合成命令：无损压缩、固定帧率、覆盖旧文件
            # ffmpeg command: lossless compression, fixed fps, overwrite old file
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # 自动覆盖已有视频文件
                "-framerate", str(self.video_fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",  # 画质参数，数值越小画质越高(0~51)
                self.video_output_name
            ]
            # 执行ffmpeg命令
            # Execute ffmpeg subprocess
            subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"仿真视频已生成：{os.path.abspath(self.video_output_name)}")
            print(f"Simulation video saved at: {os.path.abspath(self.video_output_name)}")
        finally:
            # 仿真结束自动删除临时帧文件夹
            # Auto delete temp frame folder after video generate
            shutil.rmtree(temp_dir)
            # 清空内存帧缓存释放内存
            # Clear frame cache to free memory
            self.frame_cache.clear()