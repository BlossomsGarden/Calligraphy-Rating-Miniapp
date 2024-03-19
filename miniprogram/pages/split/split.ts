import {myRequest} from '../../utils/util'
const app:IAppOption=getApp()

Page ({
  data: {
    currentImageIndex: -1, //对应手指选中了哪个字
    character: [],
    uploadBase64: '../../static/many.png',
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    this.setData({ uploadBase64: options.uploadBase64 })
    this.split()
  },

  /**
   * 调用切割接口并获得结果
   */
  split: function () {
    var that = this;
    myRequest({
      url: app.globalData.baseUrl + "split",
      data: {"data": this.data.uploadBase64},
      method: 'POST',
      modalTitle:"切割中"
    })
    .then(res=>{
      var i = 1
      var characters:Array<object> = []
      for (var url of res.scrap) {
        characters.push({ id: i, src: url })
        i += 1
      }
      that.setData({ character: characters })
      this.setData({currentImageIndex:1})
    })
    .catch(err=> {
      console.log(err)
    })
  },
  /**
   * 滑动选中书法字
   * @param event 
   */
  onFrameSelect:function(event:any) {
    this.setData({currentImageIndex:event.currentTarget.dataset.idx})
  },
  /**
   * 点击确认按钮
   */
  confirmed: function () {
    console.log(this.data.character.length)
    if(this.data.character.length==0){
      return
    }
    let selectedBase64 = this.data.character[this.data.currentImageIndex - 1].src;
    wx.redirectTo({
      url: '../style/style' + '?' + 'selectedBase64=' + selectedBase64,
    })
  },  
  /**
  * 前往我的
  */
 toMine:function(){
   wx.navigateTo({
     url: '../mine/mine',
   })
 }
})