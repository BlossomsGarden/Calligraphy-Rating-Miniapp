import {myRequest} from '../../utils/util'
const app:IAppOption=getApp()

//定义一个风格的类，用以接受推理模型的返回
interface StyleInfo {
  standard_image:string,  //所选字在该风格中的图片
  font_name:string,    //该风格的名字
  font_type:string,    //该风格的TTF文件的名字
  similarity:number,  //所选字属于该风格的可能性
}
let first3StyleList:StyleInfo[]=[]

Page({
  /**
   * 页面的初始数据
   */
  data: {
    selectedBase64: '',

    wenzi_unicode: 0,
    wenzi_ch:"识别中",

    first3StyleList:first3StyleList,
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: async function (options) {
    this.setData({ selectedBase64: options.selectedBase64 })
    await this.ocr()
    await this.getStyle()
  },

  /**
   * 汉字识别
   */
  ocr:async function(){
    await myRequest({
      url: app.globalData.baseUrl + "unicode",
      data: {"data": this.data.selectedBase64},
      method: 'POST',
      modalTitle:"识别汉字中"
    })
    .then(res=>{
      this.setData({
        wenzi_unicode:res.unicode,
        wenzi_ch:res.ch,
      })
    })
    .catch(err=> {
      console.log("OCR识别出错了",err)
    })
  },

  /**
   * 风格识别
   */
  getStyle:async function(){
    await myRequest({
      url: app.globalData.baseUrl + "style",
      data: {
        "base64": this.data.selectedBase64,
        "unicode": this.data.wenzi_unicode
      },
      method: 'POST',
      modalTitle:"识别风格中"
    })
    .then(res=>{
      this.setData({first3StyleList:res.first3TTFInfo})
      console.log("打印风格返回值", this.data.first3StyleList)
    })
    .catch(err=> {
      console.log("风格识别出错了",err)
    })
  },

  /**
   * 前往识别错误页面
   */
  gowrong:function(){
    wx.navigateTo({
      url: '../ocrerror/ocrerror'+ '?' + 'selectedBase64=' + this.data.selectedBase64,
    })
  },

  /**
   * 前往评分页面
   */
  selectStyle: function (e:any) {
    console.log("看看点击了什么", e)
    const style_name:string = this.data.first3StyleList[e.currentTarget.dataset.id].font_name
    const similarity:number = this.data.first3StyleList[e.currentTarget.dataset.id].similarity
    const demoBase64:string = this.data.first3StyleList[e.currentTarget.dataset.id].standard_image
    wx.navigateTo({
      url: '../rating/rating'+ 
      '?' + 'selectedBase64=' + this.data.selectedBase64+
      '&' + 'style=' + style_name+
      '&' + 'charmScore=' + similarity+
      '&' + 'demoBase64=' + demoBase64
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