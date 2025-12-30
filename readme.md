# 远程启动服务
nohup uvicorn main:app --host localhost --port 10093 --reload > server.log 2>&1 &

# 本地连接服务

ssh -L 10093:localhost:10093 djguo@1.94.143.93