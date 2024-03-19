import {myRequest} from '../../utils/util'
const app:IAppOption=getApp()

Page({
  /**
   * 页面的初始数据
   */
  data: {
    selectedBase64: '',
    demoBase64:'',
    score:0,
    charmScore:0,
    marks:["加载中...","加载中","加载中","加载中"],
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: async function (options:any) {
    this.setData({ selectedBase64: options.selectedBase64,})
    this.getDemoBase64(options)
    await this.getScore()
    await this.getComment()
  },
  /**
   * 获取标准字图片
   */
  getDemoBase64(options:any){
    /**
     * 这里需要补全获取到标准字图的逻辑，为了代码能跑，我只写了个demoBase64=selectedBase64
     */

    this.setData({demoBase64:options.selectedBase64})
  },
  /**
   * 回到首页
   */
  home:function(){
    wx.redirectTo({
      url: '../index/index',
    })
  },
  /**
   * 获取神韵分数和总评分
   */
  getScore:async function(){
    await myRequest({
      url: app.globalData.baseUrl + "score",
      data: {
        "selected": this.data.selectedBase64,
        "demo": this.data.demoBase64
      },
      method: 'POST',
      modalTitle:"获取评分中",
      timeout:6 * 60 * 1000,  //等待6分钟
    })
    .then(res=>{
      this.setData({
        score:res.score,
        charmScore:res.s4,
      })
    })
    .catch(err=> {
      console.log("评分出错",err)
    })
  },
  /**
   * 获取评价
   */
  getComment:async function(){
    await myRequest({
      url: app.globalData.baseUrl + "comment",
      data: {
        "selected": this.data.selectedBase64,
        "demo": this.data.demoBase64
      },
      method: 'POST',
      modalTitle:"获取修改意见中"
    })
    .then(res=>{
      this.setData({marks:res.comment})
    })
    .catch(err=> {
      console.log("评价出错",err)
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