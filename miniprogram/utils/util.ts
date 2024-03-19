//一个自带加载中icon的请求函数

interface RequestHandler {
  url: string,
  modalTitle: string,
  data?: string | object | ArrayBuffer,
  timeout?: number,
  method?: 'OPTIONS' | 'GET' | 'HEAD' | 'POST' | 'PUT' | 'DELETE' | 'TRACE' | 'CONNECT',
}

export function myRequest(requestHandler:RequestHandler) : Promise<object> {
  const data = requestHandler.data;
  const url = requestHandler.url;
  const method = requestHandler.method;
  wx.showLoading({
    title: requestHandler.modalTitle,
    mask:true
  })
  return new Promise((resolve, reject) => {
    wx.request({
      url: url,
      data: data,
      method: method,
      success: (res) => {
        wx.hideLoading();
        if(res.statusCode==200){
          console.log('myRequest: 数据原始返回。', res.data)
          resolve(res.data as object)
        }
        else{
          console.log('myRequest: 请求错误码', res.statusCode, "错误信息",res.errMsg)
          wx.showToast({
            title: '服务器维护中',
            icon:'error',
          })
          reject(res)
        }
      },
      fail: (err) => {
        wx.hideLoading();
        console.log('myRequest: 微信网络请求失败。', err)
        wx.showToast({
          title: '请刷新重试！',
          icon:'error',
        })
        reject(err)
      },
    })
  })
}
