#!/usr/bin/env python3

"""
Monitor the training progress and compare old vs new reward function
"""

import time
import os
import glob

def monitor_training():
    """Monitor the training logs and provide insights"""
    print("📊 MONITORING FIXED REWARD TRAINING")
    print("=" * 50)
    
    # Find the latest run directory
    run_dirs = glob.glob("runs/FixedFingerPush-v0__finger_push_FIXED__*")
    if not run_dirs:
        print("❌ No training runs found yet. Training may still be starting up...")
        return
    
    latest_run = max(run_dirs, key=os.path.getctime)
    print(f"📁 Latest run: {latest_run}")
    
    # Check for tensorboard events
    events_files = glob.glob(f"{latest_run}/events.out.tfevents.*")
    if events_files:
        print(f"✅ TensorBoard logs found")
        print(f"   📈 View training progress at: http://localhost:6006")
        print(f"   💡 Run: tensorboard --logdir=runs")
    
    # Check for videos
    video_dirs = glob.glob(f"videos/FixedFingerPush-v0__finger_push_FIXED__*")
    if video_dirs:
        latest_video_dir = max(video_dirs, key=os.path.getctime)
        video_files = glob.glob(f"{latest_video_dir}/*.mp4")
        print(f"🎥 Videos found: {len(video_files)} files in {latest_video_dir}")
    
    print("\n🎯 WHAT TO EXPECT WITH FIXED REWARDS:")
    print("   Old behavior: ~2388 reward from staying near target")
    print("   New behavior: Should see much lower rewards initially")
    print("   Success indicators:")
    print("     - Contact reward increasing (finger touching cube)")
    print("     - Progress reward appearing (cube moving toward target)")
    print("     - Episodes ending with success termination")
    print("     - Total episode rewards in range 50-500 (not 2000+)")
    
    print("\n📈 KEY METRICS TO WATCH:")
    print("   - Episodic return: Should be much lower but meaningful")
    print("   - Episode length: Should vary based on success/failure")
    print("   - Reward components: Contact > Distance > Success")
    
    print("\n⚡ TRAINING COMPARISON:")
    print("   Old (broken): High rewards, no actual pushing")
    print("   New (fixed): Lower rewards, actual learning expected")

if __name__ == "__main__":
    monitor_training()
