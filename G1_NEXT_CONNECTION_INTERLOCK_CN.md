# G1 下次连接强制 Interlock

**状态：COMPLETE（2026-08-28重新连接后第一项操作已验证）**

上次现场操作员离开时，LowState subscriber已经正常结束，但camera-only ZMQ server的`SIGTERM`是否成功无法通过断开的网络确认。因此机器人重新开机并恢复网络后，以下步骤被作为第一项远端操作执行。

完成证据：`results/g1_next_connection_cleanup_20260828.json`。机器人已经重启，保存的PID 8234为stale且不存活；camera、LowState和robot-command匹配进程均为0；`55555/55556/55557/60000`均无listener；没有进程被终止。`cleanup_verified=true`。该interlock已清除，但H1–H5和运动锁不因此通过。

## 第一动作：camera-only残留检查与停止

在运行任何LowState、相机probe、LGG100、IK、Shadow或故障注入之前：

1. 读取`~/g1-readonly/camera_server_readonly.pid`；
2. 如果PID存在且仍存活，验证其cmdline精确包含：
   `~/g1-readonly/g1_camera_server_readonly.py`；
3. 只有cmdline匹配时才发送`SIGTERM`；
4. 等待最多10秒；仍存活时再次记录并保持所有实验锁定，未经复核不终止其他进程；
5. 验证端口`55555/55556/55557/60000`均无listener；
6. 验证`g1_unitree_lowstate_full29.py`没有残留进程；
7. 生成cleanup JSON，只有`cleanup_verified=true`才能进入下一步。

机器人重启通常会终止上次的用户进程，但PID文件可能残留，因此仍必须检查进程身份和端口，不能仅凭重启推定已清理。

## 下次测试顺序

```text
现场Damping/机械支撑/E-stop重新确认
→ camera残留cleanup_verified
→ subscriber-only LowState smoke
→ 三相机freshness soak
→ DDS instrumented capture
→ 无Publisher Policy Shadow
```

## 不变的禁止项

```text
Publisher创建=false
模式切换=false
机器人命令=false
真机动作=false
```

在H1–H5全部通过并完成独立动作评审之前，不得运行Yuhao运动入口、MotionSwitcher、SportClient或任何命令topic。
