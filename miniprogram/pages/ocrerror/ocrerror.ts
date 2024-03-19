// pages/ocrerror/ocrerror.ts
import {myRequest} from '../../utils/util'
const app:IAppOption=getApp()
Page({

  /**
   * 页面的初始数据
   */
  data: {
      selectedBase64: '',
      inputValue:'',
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: async function (options:any) {
    this.setData({ selectedBase64: options.selectedBase64,})
  },

/**
 * 文本框键盘输入
 * 
 */
bindKeyInput: function (e:any) {
  console.log("看看输入框绑定函数",e)
  this.setData({ inputValue: e.detail.value })
},

  /**
   * 返回页面style
   */
  confirmed:function(){
    let pages = getCurrentPages();
    console.log("打印页面栈",pages)
    let prevPage = null; //上一个页面
    if (pages.length >= 2) {
      prevPage = pages[pages.length - 2]; //上一个页面
    }
    if (prevPage) {
      console.log("看看this.data.inputValue",this.data.inputValue)
      prevPage.setData({
        dataFromOcrerro: this.data.inputValue,
        wenzi_ch: this.data.inputValue
      });
    }
    wx.navigateBack({
      delta:1
    })
  },

// /**
//  * 返回上一级
//  */
// back:function(){
//   let pages = getCurrentPages();
//   console.log("打印页面栈",pages)
//   wx.navigateBack();
// },

   /**
   * 前往我的
   */
  toMine:function(){
    wx.navigateTo({
      url: '../mine/mine',
    })
  },


  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady() {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow() {

  },
 
  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide() {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload() {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh() {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom() {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage() {

  }
})