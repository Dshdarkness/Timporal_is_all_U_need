import time
import glob
import argparse
from torch.utils.data import DataLoader
# from dataset import Deepfakes
from model.TAE.sparse_dataset import Deepfakes
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from logger import prepare_logger

from utils import *  # including other libraries like torch, tqdm, etc.

import gc
gc.enable()

y_dict = {'FAKE': 0, 'REAL': 1}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_checkpoint(filepath: str):
    try:
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model = checkpoint['model']  # 提取网络结构 
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
    except Exception as e:
        log.info(e)
        # dist mode
        from multi_train_utils.distributed_utils import init_distributed_mode
        init_distributed_mode(args=args)
        checkpoint = torch.load(filepath, map_location=DEVICE)
        model = checkpoint['model']  # 提取网络结构 
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
    return model


def analysis(y_true, y_pred, y_true_frames, y_pred_frames, y_pred_prob, save_folder):
    # pre-processing
    y_true_new  = np.array([int(d) for d in y_true])
    y_pred_new  = np.array([int(d) for d in y_pred])
    y_pred_frames = np.array(y_pred_frames)

    # 混淆矩阵
    plot_confusion_matrix(np.array(y_true_frames), y_pred_frames, y_dict, save_folder)

    # 计算auc
    fpr, tpr, thresholds_keras = roc_curve(np.array(y_true_frames), y_pred_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 计算EER
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # 计算acc
    acc = metrics.accuracy_score(np.array(y_true_frames), y_pred_frames)

    log.info(f"AUC : {round(roc_auc, 3)}")
    log.info(f"ACC : {round(acc*100, 3)}")
    log.info(f"EER : {round(eer*100, 3)}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(save_folder, 'ROC_AUC.png'), format='png')

    np.save(os.path.join(save_folder, 'y_true_frames.npy'), np.array(y_true_frames))
    np.save(os.path.join(save_folder, 'y_pred_frames.npy'), y_pred_frames)
    np.save(os.path.join(save_folder, 'y_pred_prob.npy'), y_pred_prob)

    # clear memory
    del y_true_new, y_true, y_pred_new, y_pred
    gc.collect()
    return roc_auc, acc, eer


if __name__ == '__main__':
    # params settting
    parser = argparse.ArgumentParser()
    # gpu-params
    parser.add_argument('--model_name', default='Tae_wo_SRM_AE_ms_VAP_16', type=str) #test后存储的模型名字
    parser.add_argument('--data_compression', default='LQ', type=str) #测试的数据(HQ,LQ)
    parser.add_argument('--model_compression', default='LQ', type=str) #test后保存的位置HQ LQ
    args = parser.parse_args()
    # get data path
    dataset_path = f'../FF_data/FF_{args.data_compression}/train_face'
    assert os.path.exists(dataset_path), "dataset_path is not exists, plz try again."

    # obtain dataset name
    # dataset_name = ['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
    dataset_name = ['NeuralTextures'] #选择测试的数据集在这里修改

    # params
    NUM_CLASSES = 2
    SAVE_FOLDER = 'test'
    # MODEL_NAME  = 'xception'
    DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_metric = {}
    for name in dataset_name:
        # create a folder to save model
        save_folder = os.path.join(SAVE_FOLDER, args.model_compression, args.model_name, name)
        os.makedirs(save_folder, exist_ok=True)

        # create logger
        log = prepare_logger(log_dir=save_folder, log_filename="log.txt", stdout=False)

        # get dataset
        log.info(f"Now, start testing {name} dataset, using {args.model_name}...")
        # original
        test_dataset = Deepfakes(dataset_root=dataset_path, dataset_name=name, size=320,
                                 num_segments=16,frame_num=128, mode=SAVE_FOLDER, dataAug=False)

        log.info(f"Test Data Name: {name}, Test Data length: {len(test_dataset)}")

        # get dataloader
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=6,
                                 pin_memory=True,
                                 num_workers=8,
                                 shuffle=True,
                                 drop_last=True)

        # get model
        # model_save_path = glob.glob(os.path.join('weights', args.model_compression, args.model_name, name, 'best.pth'))[0]
        # GWX
        model_save_path = glob.glob(os.path.join('weights/LQ_2AE', 'Tae_wo_SRM_AE_ms_VAP_16', name, 'best.pth'))[0]
        net = load_checkpoint(model_save_path)

        # 开始测试
        y_pred, y_pred_prob, y_pred_frames, y_true, y_true_frames, y_feat = [], [], [], [], [], []
        accu_num = torch.zeros(1).to(DEVICE)  # 累计预测正确的样本数
        sample_num = 0
        with torch.no_grad():
            for step, test_data in enumerate(test_loader):
                test_images, test_labels = test_data
                sample_num += test_images.shape[0]
                outputs = net(test_images.to(DEVICE))
                # feat_out = featureNet(test_images.to(DEVICE))
                # feat_out = F.adaptive_avg_pool2d(feat_out, (1, 1)).view(30, -1)
                # y_feat.append(feat_out.to("cpu").numpy())
                pred_classes = torch.max(outputs, dim=1)[1]

                # obtain pre cls for frame level
                result_lis = pred_classes.to("cpu").numpy().tolist()

                # obtain pred cls for video level
                pred_video = np.array([0]) if result_lis.count(0) > result_lis.count(1) else np.array([1])

                # obtain pred prob for video level
                fake_prob_frames = torch.max(torch.softmax(outputs, dim=1), dim=1)[0].to("cpu").numpy().tolist()
                # fake_prob_frames = [1.0-d for d in fake_prob_frames.to("cpu").numpy()]
                # fake_prob_video  = np.mean(fake_prob_frames)
                fake_prob_frames_prob  = [1.0-d if int(result_lis[i]) == 0 else d for i,d in enumerate(fake_prob_frames)]

                # obtain ground true for frame level
                label_lis = test_labels.to("cpu").numpy().tolist()

                # obtain ground true for video level
                label_video = np.array([int(np.median(label_lis))])

                # save
                y_pred.append(pred_video)
                y_true.append(label_video)
                y_pred_frames.extend(result_lis)
                y_pred_prob.extend(fake_prob_frames_prob)
                y_true_frames.extend(label_lis)

        # do analysis for the test result
        roc_auc, acc, eer = analysis(y_true, y_pred, y_true_frames, y_pred_frames, y_pred_prob, save_folder)

        # # save embedding
        # x, y, _, = np.array(y_feat).shape
        # y_feat   = np.reshape(y_feat, (x*y, -1))
        # np.save(os.path.join(save_folder, 'y_feat.npy'), y_feat)

        dataset_metric[name] = [round(roc_auc*100, 3), round(acc*100, 3), round(eer*100, 3)]

        log.info(f'------->>>>>>Finished Testing: {args.model_name}, {name}')
        

        # clear memory
        del net, test_loader
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

    log.info(f"{dataset_metric}")
