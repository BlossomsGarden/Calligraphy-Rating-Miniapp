import {myRequest} from '../../utils/util'
const app:IAppOption=getApp()

Page({
  /**
   * 页面的初始数据
   */
  data: {
    selectedBase64: '',
    demoBase64:'',
    // 形似评分
    morphScore:{
      s1: 0,
      s2: 0,
      s3: 0,
    },
    // 神似评分
    charmScore: '获取评分中',
    // 总分
    totalScore: '获取评分中',
    marks:["加载中...","加载中...","加载中...","加载中..."],
    style:'',
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: async function (options:any) {
    this.setData({ 
      demoBase64:options.demoBase64,
      selectedBase64: options.selectedBase64,
      style:options.style,
      charmScore:Number(options.charmScore).toFixed(4)
    })
    await this.getScore()
    await this.getComment()
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
   * 计算总评分
   */
  calcTotalScore:function(){
    const totalScore:number=50*this.data.morphScore.s1 + 20*this.data.morphScore.s2 + 35*this.data.morphScore.s3+30*this.data.charmScore
    this.setData({
      totalScore: Number(totalScore).toFixed(4)
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
      timeout:2 * 60 * 1000,  //等待2分钟
    })
    .then(res=>{
      console.log("看看评分返回", res)
      this.setData({
        morphScore:res
      })
      this.calcTotalScore()
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
 * 返回上一级
 */
back:function(){
  let pages = getCurrentPages();
  console.log("打印页面栈",pages)
  wx.navigateBack();
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