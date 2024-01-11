# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter


try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


"""
# tensorboard 在 tensorflow2.1里的 生成图有问题，需要trace计算图，但不知道怎么设置pytorch model 被trace
log_dir = os.path.join(os.path.dirname(__file__), 'logs/%s' % datetime.now().strftime("%Y%m%d-%H%M%S"))
log_dir = log_dir.replace("\\", os.sep)
log_dir = log_dir.replace("/", os.sep)

tb_logger = TensorBoardLogger(log_dir,
                              trace_on=True)
        tb_logger.print_tensoroard_logs(model=model, step=epoch, loss=loss_all.item(), accuracy=accuary_all_num.item()/preds_all_num.item())
        logging.error("epoch:{} loss:{} accuracy:{}/{}={}".format(epoch, loss_all, accuary_all_num, preds_all_num,
                                                                    int(accuary_all_num)/int(preds_all_num)))
    tb_logger.write_flush()

"""
# class TensorBoardLogger(object):
#     # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
#     # https://tensorflow.google.cn/api_docs/python/tf/summary?hl=zh-CN
#     # tensorbard --logdir='./logs'
#     # https://www.cnblogs.com/rainydayfmb/p/7944224.html
#     def __init__(self, log_dir, trace_on=True):
#         """Create a summary writer logging to log_dir."""
#         self.log_dir  = log_dir
#         self.writer = tf.summary.create_file_writer(log_dir)
#         if trace_on:
#             self.trace_on()
#
#     def scalar_summary(self, tag, value, step):
#         """Log a scalar variable."""
#         with self.writer.as_default():
#             tf.summary.scalar(name=tag, data=value, step=step)
#
#
#     def image_summary(self, tag, images, step):
#         """Log a list of images."""
#
#         img_summaries = []
#         for i, img in enumerate(images):
#             # Write the image to a string
#             try:
#                 s = StringIO()
#             except:
#                 s = BytesIO()
#             scipy.misc.toimage(img).save(s, format="png")
#
#             # Create an Image object
#             img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
#                                        height=img.shape[0],
#                                        width=img.shape[1])
#             # Create a Summary value
#             img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))
#
#         # Create and write Summary
#         summary = tf.Summary(value=img_summaries)
#         self.writer.add_summary(summary, step)
#
#     def histo_summary(self, tag, values, step):
#         with self.writer.as_default():
#             tf.summary.histogram(name=tag, data=values, step=step)
#
#
#     def trace_on(self):
#         tf.summary.trace_on(
#             graph=True, profiler=True
#         )
#
#     def trace_export(self, tag, step, log_dir):
#         with self.writer.as_default():
#             tf.summary.trace_export(
#                 # tag, step=step, profiler_outdir=os.path.join(os.path.dirname(__file__), self.log_dir)
#                 tag, step=step, profiler_outdir=log_dir
#             )
#
#     def write_flush(self):
#         self.writer.flush()
#
#     def print_tensoroard_logs(self, model, step, loss, accuracy):
#         if step == 1:
#             self.trace_export(tag="graph", step=step, log_dir=self.log_dir)
#
#         def to_np(x):
#             return x.cpu().data.numpy()
#
#         info = {
#             'loss': loss,
#             'accuracy': accuracy
#         }
#
#         for tag, value in info.items():
#             self.scalar_summary(tag, value, step)
#
#         # (2) Log values and gradients of the parameters (histogram)
#         for tag, value in model.named_parameters():
#             tag = tag.replace('.', '/')
#             self.histo_summary(tag, to_np(value), step)
#             self.histo_summary(tag + '/grad', to_np(value.grad), step)
#
#         # (3) Log the images
#         # info = {
#         #     'images': to_np(img.view(-1, 28, 28)[:10])
#         # }
#         #
#         # for tag, images in info.items():
#         #     logger.image_summary(tag, images, step)

# 从tensorboardx中引入的api
class TensorBoardWritter():
    # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
    # https://tensorflow.google.cn/api_docs/python/tf/summary?hl=zh-CN
    # tensorbard --logdir='./logs'
    # https://www.cnblogs.com/rainydayfmb/p/7944224.html
    def __init__(self, log_dir=None,  comment=''):
        """Create a summary writer logging to log_dir. comment会赋在文件夹名后"""
        self.log_dir = log_dir
        if log_dir:
            self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
        else:
            self.writer = SummaryWriter(comment=comment)

    def print_tensoroard_logs(self, model, info_dict):
        sample_data = info_dict["sample_data"]
        step = info_dict["step"]
        loss = info_dict["loss"]
        loss_val = info_dict["loss_val"]
        loss_test = info_dict["loss_test"]

        accuracy = info_dict["accuracy"]
        accuary_val = info_dict["accuary_val"]
        accuary_test = info_dict["accuary_test"]
        recall_train = info_dict['recall_train']
        recall_val = info_dict['recall_val']
        recall_test = info_dict['recall_test']
        precision_train = info_dict['precision_train']
        precision_val = info_dict['precision_val']
        precision_test = info_dict['precision_test']
        F1_train = info_dict['F1_train']
        F1_val = info_dict['F1_val']
        F1_test = info_dict['F1_test']
        outputs_all = info_dict["outputs_all"]

        train_pos_neg = info_dict["train_pos_neg"]
        val_pos_neg = info_dict["val_pos_neg"]
        test_pos_neg = info_dict["test_pos_neg"]
        cross_weight_auto = info_dict["cross_weight_auto"]
        class_ac_train1, class_ac_train3, class_ac_train2 = info_dict["class_ac_train"]
        class_ac_val1, class_ac_val3, class_ac_val2 = info_dict["class_ac_val"]
        class_ac_test1, class_ac_test3, class_ac_test2 = info_dict["class_ac_test"]

        if step == 0:
            self.writer.add_graph(model, sample_data)
            self.writer.flush()

        def to_np(x):
            return x.cpu().data.numpy()

        info = {
            'loss/train': loss,
            'loss/test': loss_test,
            'loss/val': loss_val,
            'accuracy/train': accuracy,
            "accuracy/val": accuary_val,
            "accuracy/test":accuary_test,
            "class_ac/class_ac1_train1":class_ac_train1,
            "class_ac/class_ac1_val":class_ac_val1,
            "class_ac/class_ac1_test":class_ac_test1,
            "class_ac/class_ac3_train": class_ac_train3,
            "class_ac/class_ac3_val": class_ac_val3,
            "class_ac/class_ac3_test": class_ac_test3,
            "class_ac/class_ac2_train": class_ac_train2,
            "class_ac/class_ac2_val": class_ac_val2,
            "class_ac/class_ac2_test": class_ac_test2,

            "assess/precision/train": precision_train,
            "assess/precision/val": precision_val,
            "assess/precision/test": precision_test,
            "assess/recall/train": recall_train,
            "assess/recall/val": recall_val,
            "assess/recall/test": recall_test,
            "assess/F1/train": F1_train,
            "assess/F1/val": F1_val,
            "assess/F1/test": F1_test,

        }

        for tag, value in info.items():
            self.writer.add_scalar(tag, value, step)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram(tag, to_np(value), step)
            self.writer.add_histogram(tag + '/grad', to_np(value.grad), step)

        self.writer.add_histogram("outputs", to_np(outputs_all), step)
        self.writer.add_scalar("accuracy_base/train", train_pos_neg.max(), step)
        self.writer.add_scalar("accuracy_base/test", test_pos_neg.max(), step)
        self.writer.add_scalar("accuracy_base/val", val_pos_neg.max(), step)
        self.writer.add_scalar("accuracy_base/cross_weight_auto_0", cross_weight_auto[0], step)
        self.writer.add_scalar("accuracy_base/cross_weight_auto_1", cross_weight_auto[1], step)


        # (3) Log the images
        # info = {
        #     'images': to_np(img.view(-1, 28, 28)[:10])
        # }
        #
        # for tag, images in info.items():
        #     logger.image_summary(tag, images, step)


if __name__ == '__main__':
    """
    启动 tensorboard时 -log_dir=logs   -log_dir=".//logs" 
    保存graph时 需要路径分割符号为 "\"
    """
    pass
    # # writer = tf.summary.create_file_writer("./logs")
    # #
    # #
    # # @tf.function
    # # def my_func(step):
    # #     with writer.as_default():
    # #         # other model code would go here
    # #         tf.summary.scalar("my_metric", 0.5, step=step)
    # #
    # #
    # # for step in tf.range(100, dtype=tf.int64):
    # #     my_func(step)
    # #     writer.flush()
    # # The function to be traced.
    # @tf.function
    # def my_func(x, y):
    #     # A simple hand-rolled layer.
    #     return tf.nn.relu(tf.matmul(x, y))
    #
    #
    # # Set up logging.
    # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # logdir = 'logs\%s' % stamp
    # writer = tf.summary.create_file_writer(logdir)
    #
    # # Sample data for your function.
    # x = tf.random.uniform((3, 3))
    # y = tf.random.uniform((3, 3))
    #
    # # Bracket the function call with
    # # tf.summary.trace_on() and tf.summary.trace_export().
    # tf.summary.trace_on(graph=True, profiler=True)
    # # Call only one tf.function when tracing.
    # z = my_func(x, y)
    # with writer.as_default():
    #     tf.summary.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         profiler_outdir=logdir)