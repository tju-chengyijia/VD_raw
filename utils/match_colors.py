import torch 
import torch.nn.functional as F

def match_colors_ds(im_ref, im_q, im_test):
    """ Estimates a color transformation matrix between im_ref and im_q. Applies the estimated transformation to
        im_test
    """
    im_ref_ds = F.interpolate(im_ref, scale_factor=1/2, mode='bilinear')
    im_q_ds = F.interpolate(im_q, scale_factor=1/2, mode='bilinear')

    # print('im_ref =', im_ref_ds.shape, 'im_q =', im_q_ds.shape, 'im_test =', im_test.shape)
    # print(torch.max(im_ref),torch.max(im_q))
    # plt.imshow(im_ref_ds[0].permute(1,2,0).cpu())
    # plt.show()
    # plt.imshow(im_q_ds[0].permute(1,2,0).cpu())
    # plt.show()


    im_ref_mean_re = im_ref_ds.view(*im_ref_ds.shape[:2], -1)
    im_q_mean_re = im_q_ds.view(*im_q_ds.shape[:2], -1)
    # print('im_ref_mean_re =',im_ref_mean_re.shape,'im_q_mean_re =',im_q_mean_re.shape)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        # print('ir =', ir.t().shape, 'iq =', iq.t().shape)
        # print('ir =', ir.requires_grad, 'iq =', iq.requires_grad)
        c = torch.lstsq(ir.t(), iq.t())#what is the solution of least-square
        # print('c =',c.solution.requires_grad)
        # print('c_solution =',c.solution.shape)
        c = c.solution[:3]#r g b ???
        c_mat_all.append(c)

    # print('c_mat_all =',len(c_mat_all))
    c_mat = torch.stack(c_mat_all, dim=0)
    # print('c_mat =',c_mat)
    im_q_mean_conv = torch.matmul(im_q_mean_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    # print('im_q_mean_conv =',im_q_mean_conv.shape)
    im_q_mean_conv = im_q_mean_conv.view(im_q_ds.shape)
    # print('im_q_mean_conv =',im_q_mean_conv.shape)
    # plt.imshow(im_ref[0].permute(1,2,0).cpu())
    # plt.show()

    err = ((im_q_mean_conv - im_ref_ds) * 255.0).norm(dim=1)
    # print('err =',err.shape)

    # thresh = 20

    # # If error is larger than a threshold, ignore these pixels
    # valid = err < thresh
    # print('valid =',valid.shape,torch.max(valid))

    # pad = (im_q.shape[-1] - valid.shape[-1]) // 2
    # pad = [pad, pad, pad, pad]
    # valid = F.pad(valid, pad)
    # print('valid =',valid.shape,torch.max(valid))

    # upsample_factor = im_test.shape[-1] / valid.shape[-1]
    # valid = F.interpolate(valid.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear')
    # valid = valid > 0.9
    # print('valid =',valid.shape,torch.max(valid))

    # Apply the transformation to test image
    im_test_re = im_test.view(*im_test.shape[:2], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)
    # print('im_t_conv =',im_t_conv.shape)
    # plt.imshow(im_ref[0].permute(1,2,0).cpu())
    # plt.show()

    return im_t_conv#, valid


'''
if __name__ == '__main__':
    source_img = xxx
    target_img = xxx
    source_2_target = match_colors_ds(target_img,source_img,source_img)
'''