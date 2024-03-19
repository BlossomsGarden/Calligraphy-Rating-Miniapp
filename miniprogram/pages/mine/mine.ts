Component({
  data: {
    userInfo: {
      avatarUrl: "../../static/profile.png",
      nickName: "未登录",
    },
    hasUserInfo: false,
    canIUseGetUserProfile: wx.canIUse('getUserProfile'),
    canIUseNicknameComp: wx.canIUse('input.type.nickname'),
  },
  methods: {
    onShow:function(){
      const that=this
      wx.getStorage({
        key: 'userInfo',
        success: function(res) {
          console.log('从缓存取userInfo成功',res)
          that.setData({
            userInfo: {avatarUrl:res.data.avatarUrl,nickName:res.data.nickName},
            hasUserInfo:true
          })
        },
      })
    },

    back(){
      let pages = getCurrentPages();
      console.log("打印页面栈",pages)
      wx.navigateBack();
    },

    getUserProfile(){
      if(this.data.hasUserInfo){
        wx.showToast({ title: '已登录！',icon:"none", duration: 2000 })
        return
      }
      wx.getUserProfile({
        desc: '登录', 
        success: (res) => {
          this.setData({
            userInfo:{nickName: res.userInfo.nickName,avatarUrl: res.userInfo.avatarUrl},
            hasUserInfo:true
          })
          wx.setStorage({
            key: 'userInfo',
            data: this.data.userInfo,
            success: function() {
              console.log('userInfo写入缓存成功')
            },
          })
        }
      })
    }
  },
})
