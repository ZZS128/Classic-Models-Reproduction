# Classic-Models-Reproduction
based on:https://www.bilibili.com/video/BV1e34y1M7wR

#### LeNet-5 报错及解决
1. "Git: fatal: unable to access 'xxx.git': Failed to connect to github.com port 443 after 21064 ms: Could not connect to server"
> 挂全局梯，无用。后发现将git端口号从443修改为代理端口号即可解决
2. "OSError[WinError 1455]页面文件太小，无法完成操作:xxx."
> dataloader中的num_workers设置的过大导致内存爆炸，调整为4即可。
3. "TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
> pandas想将GPU中的张量转化为Numpy多维数组时出错，因为Numpy只能操作CPU上的张量。可采取.cpu()/.item()/.tolist()解决
