import lpips, os, math, cv2, torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)

    return composed_transform(img)
# def init():
#     # Make dirs
#     #mkdir(args.TEST_RESULT_DIR)

#     #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.GPU_ID
#     #logger = SummaryWriter(args.LOGS_DIR)

#     # initialize lpips
#     net_metric_alex = lpips.LPIPS(net='alex').cuda()

#     # random seed
#     # random.seed(args.SEED)
#     # np.random.seed(args.SEED)
#     # torch.manual_seed(args.SEED)
#     # torch.cuda.manual_seed_all(args.SEED)
#     # if args.SEED == 0:
#     #     torch.backends.cudnn.deterministic = True
#     #     torch.backends.cudnn.benchmark = False
#     # else:
#     #     torch.backends.cudnn.deterministic = False
#     #     torch.backends.cudnn.benchmark = True
    
#     return net_metric_alex

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.set_num_threads(16)

gt_path = '/ssd/cyj/video_dataset/merge/test/gt_rgb/'
# gt_path = '/ssd/cyj/raw_moire_image_dataset/testset/gt_RGB/'
dm_path = '/data/cyj/VD_raw/test_result/0058/'
log_path = '/data/cyj/VD_raw/test_result/0058_'
#Mechanical hard disk '/media/storage/cyj/vedio_demoireing/EDVR-master/edvr_result/raw_300000iter/'
#SSD '/media/data/cyj/edvr_result/raw_300000iter/'

gt_file = sorted(os.listdir(gt_path))
num_file = len(gt_file)
print('The number of files is',)

net_metric = lpips.LPIPS(net='alex').cuda()

f = open(log_path + 'result.txt', "w")
avg_lpips, avg_psnr, avg_ssim = 0, 0, 0
with torch.no_grad():
    for i in gt_file:
        print(i)

        gt_file = gt_path + i
        # output_file = dm_path + 'test_' + i
        # output_file = dm_path + i.split('_')[0] + '_dm.png'
        output_file = dm_path + 'val_' + i[:5] + '000' + i[5:]
        # output_file = dm_path + i[1:4] + '/' + i
        # output_file = dm_path + 'val_tensor(' + i.split('_')[0] + ').png'

        gt = Image.open(gt_file).convert('RGB')
        gt_tensor = default_toTensor(gt).cuda()
        gt = gt_tensor.cpu().numpy().transpose(1,2,0)*255
        output = Image.open(output_file).convert('RGB')
        output_tensor = default_toTensor(output).cuda()
        output = output_tensor.cpu().numpy().transpose(1,2,0)*255

        # Calculate LPIPS
        cur_lpips = net_metric.forward(output_tensor, gt_tensor, normalize=True)
        avg_lpips += cur_lpips
        # Calculate PSNR
        cur_psnr = calculate_psnr(output, gt)
        avg_psnr += cur_psnr
        # Calculate SSIM
        cur_ssim = calculate_ssim(output, gt)
        avg_ssim += cur_ssim
        f.write('%06s: LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f \n' % (i, cur_lpips, cur_psnr, cur_ssim))
        print('%06s: LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f \n' % (i, cur_lpips, cur_psnr, cur_ssim))
        # del gt_tensor, output_tensor, cur_lpips
        # torch.cuda.empty_cache()
avg_lpips = avg_lpips/num_file
avg_psnr = avg_psnr/num_file
avg_ssim = avg_ssim/num_file
f.write('Avg. LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f \n' % (avg_lpips, avg_psnr, avg_ssim))
print('Avg. LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f \n' % (avg_lpips, avg_psnr, avg_ssim))
f.close()