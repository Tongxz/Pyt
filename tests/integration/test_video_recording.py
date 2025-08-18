#!/usr/bin/env python3
"""
测试视频录制功能
"""

import os

import requests


def test_video_recording():
    # API端点
    url = "http://localhost:8000/api/v1/detect/comprehensive"

    # 查找测试视频文件
    test_video_paths = [
        "tests/fixtures/videos/test_video.mp4",
        "tests/fixtures/videos/sample.mp4",
        "tests/fixtures/videos/test.mp4",
    ]

    test_video = None
    for path in test_video_paths:
        if os.path.exists(path):
            test_video = path
            break

    if not test_video:
        print("❌ 没有找到测试视频文件")
        print("请在以下路径之一放置测试视频:")
        for path in test_video_paths:
            print(f"  - {path}")
        print("\n💡 提示: 您可以使用任何MP4视频文件进行测试")
        print("或者直接在浏览器中测试:")
        print("1. 打开 http://localhost:8000")
        print("2. 选择'视频检测'")
        print("3. 勾选'📹 录制检测过程'")
        print("4. 上传视频文件进行检测")
        return

    print(f"📹 使用测试视频: {test_video}")

    # 测试不录制模式
    print("\n🔍 测试1: 不录制模式")
    with open(test_video, "rb") as f:
        files = {"file": (os.path.basename(test_video), f, "video/mp4")}
        data = {"record_process": "false"}

        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print("✅ 检测成功")
            print(f"   检测到人数: {result.get('statistics', {}).get('total_persons', 0)}")
            if "output_video" in result:
                print("❌ 错误: 不录制模式下不应该有output_video")
            else:
                print("✅ 正确: 不录制模式下没有output_video")
        else:
            print(f"❌ 检测失败: {response.status_code}")
            print(response.text)

    # 测试录制模式
    print("\n🎬 测试2: 录制模式")
    with open(test_video, "rb") as f:
        files = {"file": (os.path.basename(test_video), f, "video/mp4")}
        data = {"record_process": "true"}

        response = requests.post(url, files=files, data=data)

        if response.status_code == 200:
            result = response.json()
            print("✅ 检测成功")
            print(f"   检测到人数: {result.get('statistics', {}).get('total_persons', 0)}")

            if "output_video" in result:
                output_info = result["output_video"]
                print("✅ 成功生成录制视频:")
                print(f"   文件名: {output_info.get('filename')}")
                print(f"   文件大小: {output_info.get('size_bytes', 0)} bytes")
                print(f"   下载URL: {output_info.get('url')}")

                # 检查文件是否真的存在
                video_path = output_info.get("path")
                if video_path and os.path.exists(video_path):
                    print(f"✅ 视频文件已生成: {video_path}")
                else:
                    print(f"❌ 视频文件不存在: {video_path}")
            else:
                print("❌ 错误: 录制模式下应该有output_video")

            if "processing_info" in result:
                proc_info = result["processing_info"]
                print("📊 处理信息:")
                print(f"   总帧数: {proc_info.get('total_frames')}")
                print(f"   处理帧数: {proc_info.get('processed_frames')}")
                print(f"   处理时间: {proc_info.get('processing_time', 0):.2f}秒")
        else:
            print(f"❌ 检测失败: {response.status_code}")
            print(response.text)


if __name__ == "__main__":
    test_video_recording()
