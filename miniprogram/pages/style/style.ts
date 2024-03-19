Page({
  /**
   * 页面的初始数据
   */
  data: {
    selectedBase64: '',
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    this.setData({ selectedBase64: options.selectedBase64 })
  },
  
  /**
   * 前往评分页面
   */
  gotoPage: function () {
    wx.navigateTo({
      url: '../rating/rating'+ '?' + 'selectedBase64=' + this.data.selectedBase64,
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