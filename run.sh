curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.bashrc  

nvm install 22
nvm use 22

npm install -g pnpm
pnpm install
pnpm ui:build 
pnpm build


pnpm openclaw onboard --flow quickstart   

#server side
pnpm openclaw gateway run --bind loopback --port 18789 


#client side
pnpm openclaw tui

#-------------- INT8  -----------------
# /msdl Qwen/Qwen3.0-0.6B
# /msdl  Qwen/Qwen-Image-Edit-2511 https://huggingface.co/Qwen/Qwen-Image-Edit-2511

#-------------- INT4  -----------------

# MiniMaxAI/MiniMax-M2.1
#https://huggingface.co/Qwen/Qwen3-235B-A22B
#https://huggingface.co/deepseek-ai/DeepSeek-V3.2
#https://huggingface.co/MiniMaxAI/MiniMax-M2.1

