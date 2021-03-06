import torch

# define model
class ResNet_50(torch.nn.Module):
    # block 1
    def block_1_1(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

    def block_1_2(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

    def block_1_3(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256)
        )    
        
    # block 2
    def block_2_init_1(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 128, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )

    def block_2_init_2(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )

    def block_2_init_3(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(128, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512)
        )
    
    def block_2_1(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )

    def block_2_2(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True)
        )

    def block_2_3(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(128, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512)
        )
    
    # block 3
    def block_3_init_1(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 256, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )

    def block_3_init_2(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )

    def block_3_init_3(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(256, 1024, kernel_size=1),
            torch.nn.BatchNorm2d(1024)
        )
    
    def block_3_1(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )

    def block_3_2(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )

    def block_3_3(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(256, 1024, kernel_size=1),
            torch.nn.BatchNorm2d(1024)
        )
    
    # block 4
    def block_4_init_1(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 512, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )

    def block_4_init_2(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )

    def block_4_init_3(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(512, 2048, kernel_size=1),
            torch.nn.BatchNorm2d(2048)
        )
    
    def block_4_1(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_features, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )

    def block_4_2(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(inplace=True)
        )

    def block_4_3(self, in_features):
        return torch.nn.Sequential(
            torch.nn.Conv2d(512, 2048, kernel_size=1),
            torch.nn.BatchNorm2d(2048)
        )
    
    
    # init function
    def __init__(self, num_classes):
        super(ResNet_50, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        
        self.pool = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # spatial attention
        self.spatial_attention = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=7, padding=3, stride=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid()
        )

        # channel attention
        self.max_pool_11 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=112, stride=112))
        self.max_pool_1 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=56, stride=56))
        self.max_pool_2 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=28, stride=28))
        self.max_pool_3 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=14, stride=14))
        self.max_pool_4 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=7, stride=7))
        self.avg_pool_11 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=112, stride=112))
        self.avg_pool_1 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=56, stride=56))
        self.avg_pool_2 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=28, stride=28))
        self.avg_pool_3 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=14, stride=14))
        self.avg_pool_4 = torch.nn.Sequential(torch.nn.AvgPool2d(kernel_size=7, stride=7))

        # block 1
        self.resnet_block_1_1_1 = self.block_1_1(64)
        self.resnet_block_1_1_2 = self.block_1_2(64)
        self.resnet_block_1_1_3 = self.block_1_3(64)
        self.resnet_block_1_2_1 = self.block_1_1(256)
        self.resnet_block_1_2_2 = self.block_1_2(256)
        self.resnet_block_1_2_3 = self.block_1_3(256)
        self.resnet_block_1_3_1 = self.block_1_1(256)
        self.resnet_block_1_3_2 = self.block_1_2(256)
        self.resnet_block_1_3_3 = self.block_1_3(256)
        
        # block 2
        self.resnet_block_2_1_1 = self.block_2_init_1(256)
        self.resnet_block_2_1_2 = self.block_2_init_2(256)
        self.resnet_block_2_1_3 = self.block_2_init_3(256)
        self.resnet_block_2_2_1 = self.block_2_1(512)
        self.resnet_block_2_2_2 = self.block_2_2(512)
        self.resnet_block_2_2_3 = self.block_2_3(512)
        self.resnet_block_2_3_1 = self.block_2_1(512)
        self.resnet_block_2_3_2 = self.block_2_2(512)
        self.resnet_block_2_3_3 = self.block_2_3(512)
        self.resnet_block_2_4_1 = self.block_2_1(512)
        self.resnet_block_2_4_2 = self.block_2_2(512)
        self.resnet_block_2_4_3 = self.block_2_3(512)
        
        # block 3
        self.resnet_block_3_1_1 = self.block_3_init_1(512)
        self.resnet_block_3_1_2 = self.block_3_init_2(512)
        self.resnet_block_3_1_3 = self.block_3_init_3(512)
        self.resnet_block_3_2_1 = self.block_3_1(1024)
        self.resnet_block_3_2_2 = self.block_3_2(1024)
        self.resnet_block_3_2_3 = self.block_3_3(1024)
        self.resnet_block_3_3_1 = self.block_3_1(1024)
        self.resnet_block_3_3_2 = self.block_3_2(1024)
        self.resnet_block_3_3_3 = self.block_3_3(1024)
        self.resnet_block_3_4_1 = self.block_3_1(1024)
        self.resnet_block_3_4_2 = self.block_3_2(1024)
        self.resnet_block_3_4_3 = self.block_3_3(1024)
        self.resnet_block_3_5_1 = self.block_3_1(1024)
        self.resnet_block_3_5_2 = self.block_3_2(1024)
        self.resnet_block_3_5_3 = self.block_3_3(1024)
        self.resnet_block_3_6_1 = self.block_3_1(1024)
        self.resnet_block_3_6_2 = self.block_3_2(1024)
        self.resnet_block_3_6_3 = self.block_3_3(1024)
        
        # block 4
        self.resnet_block_4_1_1 = self.block_4_init_1(1024)
        self.resnet_block_4_1_2 = self.block_4_init_2(1024)
        self.resnet_block_4_1_3 = self.block_4_init_3(1024)
        self.resnet_block_4_2_1 = self.block_4_1(2048)
        self.resnet_block_4_2_2 = self.block_4_2(2048)
        self.resnet_block_4_2_3 = self.block_4_3(2048)
        self.resnet_block_4_3_1 = self.block_4_1(2048)
        self.resnet_block_4_3_2 = self.block_4_2(2048)
        self.resnet_block_4_3_3 = self.block_4_3(2048)
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d(7)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2048 * 7 * 7, num_classes)
        )
        
        self.relu = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True)
        )
        
        self.skip_connection_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256)
        )
        
        self.skip_connection_2 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(512)
        )
        
        self.skip_connection_3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 1024, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(1024)
        )
        
        self.skip_connection_4 = torch.nn.Sequential(
            torch.nn.Conv2d(1024, 2048, kernel_size=1, stride=2),
            torch.nn.BatchNorm2d(2048)
        )
        
        
    # define forward function
    def forward(self, x):
        
        # apply initial conv layers
        x = self.features(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_11(x) + self.avg_pool_11(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.pool(x)
        
        # block 1
        input = x
        x = self.resnet_block_1_1_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_1_1_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_1_1_3(x)
        input = self.skip_connection_1(input)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_1_2_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_1_2_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_1_2_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_1_3_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_1_3_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_1_3_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_1(x) + self.avg_pool_1(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        
        # block 2
        input = x
        x = self.resnet_block_2_1_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_2_1_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_2_1_3(x)
        input = self.skip_connection_2(input)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_2_2_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_2_2_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_2_2_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_2_3_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_2_3_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_2_3_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_2_4_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_2_4_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_2_4_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_2(x) + self.avg_pool_2(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        
        # block 3
        input = x
        x = self.resnet_block_3_1_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_3_1_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_3_1_3(x)
        input = self.skip_connection_3(input)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_3_2_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_3_2_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_3_2_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_3_3_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_3_3_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_3_3_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_3_4_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_3_4_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_3_4_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_3_5_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_3_5_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_3_5_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_3_6_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_3_6_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_3_6_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_3(x) + self.avg_pool_3(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        
        # block 4
        input = x
        x = self.resnet_block_4_1_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_4_1_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_4_1_3(x)
        input = self.skip_connection_4(input)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_4_2_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_4_2_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_4_2_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        input = x
        x = self.resnet_block_4_3_1(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x = self.resnet_block_4_3_2(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        x_1 = self.resnet_block_4_3_3(x)
        x = torch.add(input, x_1)
        x = self.relu(x)
        scale = torch.nn.functional.sigmoid(self.max_pool_4(x) + self.avg_pool_4(x)).expand_as(x)
        x = x * scale
        scale = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = self.spatial_attention(scale)
        x = x * scale
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x