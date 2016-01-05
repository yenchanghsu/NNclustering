local mnist = require 'mnist'

local trainset = mnist.traindataset()
trainset.data = trainset.data:float()
local mean = trainset.data:mean()
local stdv = trainset.data:std()
trainset.data:add(-mean)
trainset.data:div(stdv)
trainset.data=trainset.data:view(torch.LongStorage{-1,1,28,28}) -- add the dimension of channel
trainset.label:maskedFill(trainset.label:eq(0),10) -- replace the label 0 with 10 to avoid index issue in ClassNLLCriterion

function trainset:size() 
    return self.data:size(1)
end
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);

local testset = mnist.testdataset()
testset.data = testset.data:float()
testset.data:add(-mean)
testset.data:div(stdv)
testset.data=testset.data:view(torch.LongStorage{-1,1,28,28})   -- add the dimension of channel even it is 1
testset.label:maskedFill(testset.label:eq(0),10)

provider = {}
provider.trainData = {}
provider.testData={}

provider.trainData.data = trainset.data
provider.trainData.labels = trainset.label
provider.testData.data = testset.data
provider.testData.labels = testset.label