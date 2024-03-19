const app:IAppOption=getApp()
Page({

  /**
   * 页面的初始数据
   */
  data: {
    uploadBase64:'',
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad() {

  },

  uploadAction: function (event: any) {
    // 获取通过data-*传递的参数
    const type = event.currentTarget.dataset.type;
    let that = this;
    wx.chooseMedia({
      count: 1, // 最多可选择的文件数（默认9）
      mediaType: ['image'],
      sizeType: ['original'], // 可以指定是原图还是压缩图，默认二者都有
      sourceType: ['album', 'camera'], // 可以指定来源是相册还是相机，默认二者都有
      success(res1) {
        let filePath = res1.tempFiles[0].tempFilePath
        let thatT = that
        wx.showLoading({
          title: '上传中',
        })
        wx.uploadFile({
          filePath: filePath, //临时文件路径
          url: app.globalData.baseUrl + 'upload',  //填写服务器接口的访问地址
          name: 'imgs',  //name对应接口所需要传递的图片的key
          timeout: 5000,
          success(res2) {
            wx.hideLoading()
            //目前小程序获取不到图片本地的完整名称,都是临时文件路径，因此是乱码
            console.log("上传成功",res2)
            thatT.setData({ uploadBase64: JSON.parse(res2.data).base64 })
            // 根据type的值决定接下来的操作
            if (type === 'single') {
              console.log('上传单字，直接识别');
              wx.redirectTo({
                url: '../style/style' + '?' + 'selectedBase64=' + that.data.uploadBase64,
              })
            } 
            else if (type === 'practice') {
              console.log('上传习作，跳转切割页面');
              wx.redirectTo({
                url: '../split/split' + '?' + 'uploadBase64=' + that.data.uploadBase64,
              })
            }
          },
          fail(res3){
            console.log("失败",res3)
            wx.hideLoading()
            wx.showToast({ title: '上传失败，请重试',icon:"none", duration: 2000 })
          }
        })
      }
    })
  },

  toMine:function(){
    wx.navigateTo({
      url: '../mine/mine',
    })
  }
})