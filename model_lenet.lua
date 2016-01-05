require 'nn'
require 'cunn'
local backend_name = opt.backend

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end

local net = nn.Sequential()

net = nn.Sequential()

net:add(backend.SpatialConvolution(im_channel, 20, 5, 5))
net:add(nn.SpatialBatchNormalization(20,1e-3))
net:add(backend.ReLU(true))
net:add(backend.SpatialMaxPooling(2,2,2,2))

net:add(backend.SpatialConvolution(20, 50, 5, 5))
net:add(nn.SpatialBatchNormalization(50,1e-3))
net:add(backend.ReLU(true))
net:add(backend.SpatialMaxPooling(2,2,2,2))

net:add(nn.View(50*after_pool_sz*after_pool_sz))
net:add(nn.Linear(50*after_pool_sz*after_pool_sz, 500))
net:add(nn.BatchNormalization(500))
net:add(nn.ReLU(true))
net:add(nn.Linear(500, 10))

return net