import argparse


def cfg(data_set):
    parser = argparse.ArgumentParser(description='AAAF')

    parser.add_argument('--train_raw_path', type=str,
                        default=f'D:\\dataset\\{data_set}\\train_raw/',
                        help='训练原始图像路径')
    parser.add_argument('--train_gt_path', type=str,
                        default=f'D:\\dataset\\{data_set}\\train_GT/',
                        help='训练真实图像路径')
    parser.add_argument('--test_raw_path', type=str,
                        default=f'D:\\dataset\\{data_set}\\test_raw/',
                        help='测试原始图像路径')
    parser.add_argument('--test_gt_path', type=str,
                        default=f'D:\\dataset\\{data_set}\\test_GT/',
                        help='测试真实图像路径')
    parser.add_argument('--model_save_dir', type=str,
                        default=f'saved_models/{data_set}',
                        help='模型保存目录')
    parser.add_argument('--output_path', type=str,
                        default=f'./output/{data_set}',
                        help='图像结果输出路径')
    parser.add_argument('--saved_models_path', type=str,
                        default=f'saved_models/{data_set}',
                        help='模型保存目录')

    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--n', type=int, default=2, help="模型规模")

    parser.add_argument('--batch_size', type=int, default=8,
                        help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练总轮次')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--step_size', type=int, default=25,
                        help='学习率调度步长')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='学习率衰减因子')

    parser.add_argument('--lambda_pixel', type=float, default=1,
                        help='像素损失权重')
    parser.add_argument('--lambda_ssim', type=float, default=0.2,
                        help='SSIM损失权重')
    parser.add_argument('--lambda_vgg', type=float, default=0.5,
                        help='颜色损失权重')

    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help='检查点打印间隔')
    parser.add_argument('--checkpoint_save_interval', type=int, default=2,
                        help='模型保存间隔')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像尺寸')


    parser.add_argument('--cuda_device', type=str, default='0',
                        help='CUDA设备ID')

    return parser.parse_args()



