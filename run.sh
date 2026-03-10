pnpm install
pnpm ui:build 
pnpm build


pnpm openclaw onboard --flow quickstart   

#server side
pnpm openclaw gateway run --bind loopback --port 18789 


#client side
pnpm openclaw tui
# /msdl Qwen/Qwen3.5-0.8B