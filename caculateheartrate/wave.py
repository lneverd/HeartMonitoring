import pywt
import numpy as np
import matplotlib.pyplot as plt
def smallwave(data):


    w = pywt.Wavelet('db8')
    coeffs = pywt.wavedec(data=data, wavelet=w, level=4)
    # y = pywt.waverec(c,i,'db8') 信号重构
    # [cA_n, cD_n, cD_n - 1, ..., cD2, cD1]: list
    # cd3、4、5、6、
    cA4, cD4, cD3, cD2, cD1 = coeffs
    # print('CA4', cA4, len(cA4))
    # print('CD4', cD4, len(cD4))
    # print('CD3', cD3, len(cD3))
    # print('CD2', cD2, len(cD2))
    # print('CD1', cD1, len(cD1))
    coeffs1 = [np.zeros(len(cA4)),cD4, cD3,np.zeros(len(cD2)),np.zeros(len(cD1))]
    # print(len(coeffs))
    s1 = pywt.waverec(coeffs1, w)
    return s1

def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))
def test():
    data = np.loadtxt('ppg.txt')
    coeffs = pywt.wavedec(data, 'db8', level=6);
    print(len(coeffs))



if __name__ == '__main__':

#     x = [1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20]
#     wp = pywt.WaveletPacket(data=x, wavelet='db8', mode='symmetric',maxlevel=6)
#     print(wp.data) #输入的数据
# #     print(repr(wp.path)) #标识节点路径
# #     print(wp['ad'].maxlevel) #关于最大分解层数，如果构造函数中没有指定参数，则自动计算。
# # #     获取小波包树的子节点
# # #     1st level
# #     print(wp['a'].data)
# #
# #     print(wp['a'].path)
# # #     2nd level
# #     print(wp['aa'].data)
# #     print(wp['aa'].path)
# #     #3rd level
# #     print(wp['aaa'].data)
# #     print(wp['aaa'].path)
# #
# #     print(wp['aaaa'].data)
# #     print(wp['aaaa'].path)
#
#     # print(wp['ac']) 错误路径
#
# #     访问子节点
#     print('访问子节点:',wp['ad'].data)
#     print(wp['ad'].path)
#     print(wp['ad'].node_name)
#     print(wp['ad'].parent.path)
#     print(wp['ad'].level)
#     print(wp['ad'].maxlevel)
#     print(wp['ad'].mode)
#
#     # for node in wp.get_level(6,'natural'):
#     #     print(node.path)
#     print([node.path for node in wp.get_level(6, 'natural')])
#
#     x = [1, 2, 3, 4, 5, 6, 7, 8]
#     wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
#     new_wp = pywt.WaveletPacket(data=None, wavelet='db1', mode='symmetric')
#     new_wp['aa'] = wp['aa'].data
#     new_wp['ad'] = [-2., -2.]
#     new_wp['d'] = wp['d']
#     print(new_wp.reconstruct(update=False))
#     print([n.path for n in new_wp.get_leaf_nodes(False)])
#     # 如果reconstruct方法中的update参数被设置为False，那么根节点的数据将不会被更新。
#     print(new_wp.data)
#     print(new_wp.reconstruct(update=True))
#     print([n.path for n in new_wp.get_leaf_nodes(True)])
#     print(new_wp.data)
#
    x = [1, 2, 3, 4, 5, 6, 7, 8]
#     wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
#     dummy = wp.get_level(2)
#     for n in wp.get_leaf_nodes(False):
#         print(n.path, n.data)
#
#     del wp['ad'] #删除一个节点
#     for n in wp.get_leaf_nodes(False):
#         print(n.path, n.data)
#
    # 小波分解
    data = [3,7,1,1,-2,5,4,6,6,4,5,-2,1,1,7,3]
    coeffs = pywt.wavedec(data,'db8',level=2,mode='periodic')
    cA2,cD2,cD1 = coeffs
    y = pywt.waverec(coeffs,'db8',mode='periodic')
    print('CA2',cA2,len(cA2))
    print('CD2',cD2,len(cD2))
    print('CD1',cD1,len(cD1))

