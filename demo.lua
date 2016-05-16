require 'xlua'
require 'optim'
require 'nn'
require 'BatchKLDivCriterion.lua'
require 'hungarian'
local c = require 'trepl.colorize'

opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)         batch size
   -r,--learningRate          (default 0.1)         learning rate
   --learningRateDecay        (default 1e-7)        learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --max_epoch                (default 100)         maximum number of iterations
   -d,--dataset               (default "mnist")     mnist or cifar10
   --backend                  (default "cudnn")     nn (for cpu only), cunn, cudnn (fastest)
   --clustering               (default 0)
]]

print(opt)

print(c.blue '==>' ..' loading data')
if opt.dataset=="cifar10" then
  dofile('provider_cifar10.lua')
  local f=io.open('provider_cifar10.t7','r')
  if f~=nil then
    io.close(f)
    print('Load CIFAR-10 cache file...')
    provider = torch.load('provider_cifar10.t7')
  else
    print('Creating CIFAR-10 cache file...')
    provider = Provider()
    provider:normalize()
    torch.save('provider_cifar10.t7',provider)
  end
  im_channel = 3
  after_pool_sz = 5
else
  dofile('provider_mnist.lua')
  im_channel = 1
  after_pool_sz = 4
end

print(c.blue '==>' ..' configuring model')
model = dofile('model_lenet.lua')

print(c.blue'==>' ..' setting criterion')
if opt.clustering==1 then
  model:add(nn.SoftMax())
  criterion = nn.BatchKLDivCriterion(2)
else
  criterion = nn.CrossEntropyCriterion():float()
end

if opt.backend~="nn" then
  model = model:cuda()
  model:add(nn.Copy('torch.CudaTensor','torch.FloatTensor'))
else
  model = model:float()
end
print(model,criterion)

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = torch.FloatTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)

    local inputs = provider.trainData.data:index(1,v)
    if opt.backend~="nn" then inputs=inputs:cuda() end
    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      if opt.backend~="nn" then df_do=df_do:cuda() end
      model:backward(inputs, df_do)

      confusion:batchAdd(outputs, targets)

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end
  
  -- assign each cluster to the dominate label
  if opt.clustering==1 then
    ind = hungarian.maxCost(confusion.mat:int():t())
    ind = ind:squeeze()
    new_mat = confusion.mat:clone():fill(0)
    for i=1,ind:nElement() do
      new_mat[i] = confusion.mat[ind[i] ]
    end
    confusion.mat:copy(new_mat)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 100
  for i=1,provider.testData.data:size(1),bs do
    local inputs = provider.testData.data:narrow(1,i,bs)
    if opt.backend~="nn" then inputs=inputs:cuda() end
    local outputs = model:forward(inputs)
    confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
  end
  
  -- optimal assigment (based on training set)
  if opt.clustering==1 then
    new_mat = confusion.mat:clone():fill(0)
    for i=1,ind:nElement() do
      new_mat[i] = confusion.mat[ind[i] ]
    end
    confusion.mat:copy(new_mat)
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    local base64im
    do
      os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
      os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
      local f = io.open(opt.save..'/test.base64')
      if f then base64im = f:read'*all' end
    end

    local file = io.open(opt.save..'/report.html','w')
    file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save,epoch,base64im))
    for k,v in pairs(optimState) do
      if torch.type(v) == 'number' then
        file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
      end
    end
    file:write'</table><pre>\n'
    file:write(tostring(confusion)..'\n')
    file:write(tostring(model)..'\n')
    file:write'</pre></body></html>'
    file:close()
  end

  -- save model every 50 epochs
  if epoch % 50 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename, model:get(3))
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  train()
  test()
end


