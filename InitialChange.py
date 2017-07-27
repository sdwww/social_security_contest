def single_get_first(unicode1):
    str1 = unicode1.encode('gbk')
    try:
        ord(str1)
        return str1
    except:
        asc = str1[0] * 256 + str1[1] - 65536
        if asc >= -20319 and asc <= -20284:
            return 'a'
        if asc >= -20283 and asc <= -19776:
            return 'b'
        if asc >= -19775 and asc <= -19219:
            return 'c'
        if asc >= -19218 and asc <= -18711:
            return 'd'
        if asc >= -18710 and asc <= -18527:
            return 'e'
        if asc >= -18526 and asc <= -18240:
            return 'f'
        if asc >= -18239 and asc <= -17923:
            return 'g'
        if asc >= -17922 and asc <= -17418:
            return 'h'
        if asc >= -17417 and asc <= -16475:
            return 'j'
        if asc >= -16474 and asc <= -16213:
            return 'k'
        if asc >= -16212 and asc <= -15641:
            return 'l'
        if asc >= -15640 and asc <= -15166:
            return 'm'
        if asc >= -15165 and asc <= -14923:
            return 'n'
        if asc >= -14922 and asc <= -14915:
            return 'o'
        if asc >= -14914 and asc <= -14631:
            return 'p'
        if asc >= -14630 and asc <= -14150:
            return 'q'
        if asc >= -14149 and asc <= -14091:
            return 'r'
        if asc >= -14090 and asc <= -13119:
            return 's'
        if asc >= -13118 and asc <= -12839:
            return 't'
        if asc >= -12838 and asc <= -12557:
            return 'w'
        if asc >= -12556 and asc <= -11848:
            return 'x'
        if asc >= -11847 and asc <= -11056:
            return 'y'
        if asc >= -11055 and asc <= -10247:
            return 'z'
        return ''


def getPinyin(string):
    if string == None:
        return None
    lst = list(string)
    charLst = []
    for l in lst:
        charLst.append((single_get_first(l)).upper())
    return ''.join(charLst)


if __name__ == '__main__':
    items = '顺序号,个人编码,医院编码,药品费发生金额,贵重药品发生金额,中成药费发生金额,中草药费发生金额,药品费自费金额,' \
            '药品费拒付金额,药品费申报金额,检查费发生金额,贵重检查费金额,检查费自费金额,检查费拒付金额,检查费申报金额,' \
            '治疗费发生金额,治疗费自费金额,治疗费拒付金额,治疗费申报金额,手术费发生金额,手术费自费金额,手术费拒付金额,' \
            '手术费申报金额,床位费发生金额,床位费拒付金额,床位费申报金额,医用材料发生金额,高价材料发生金额,' \
            '医用材料费自费金额,医用材料费拒付金额,输全血申报金额,成分输血自费金额,成分输血拒付金额,成分输血申报金额,' \
            '其它发生金额,其它拒付金额,其它申报金额,一次性医用材料自费金额,一次性医用材料拒付金额,一次性医用材料申报金额,' \
            '输全血按比例自负金额,起付线标准金额,起付标准以上自负比例金额,医疗救助个人按比例负担金额,最高限额以上金额,' \
            '统筹拒付金额,基本医疗保险统筹基金支付金额,交易时间,农民工医疗救助计算金额,公务员医疗补助基金支付金额,' \
            '城乡救助补助金额,可用账户报销金额,基本医疗保险个人账户支付金额,非账户支付金额,双笔退费标识,住院开始时间,' \
            '住院终止时间,住院天数,申报受理时间,出院诊断病种名称,本次审批金额,补助审批金额,医疗救助医院申请,' \
            '残疾军人医疗补助基金支付金额,民政救助补助金额,城乡优抚补助金额,非典补助补助金额,家床起付线剩余,操作时间,' \
            '三目统计项目,三目服务项目名称,三目医院服务项目名称,剂型,规格,单价,数量,拒付原因编码,' \
            '拒付原因,费用发生时间'
    each_item=items.split(',')
    new_each_item=[]
    for i in each_item:
        new_each_item.append(getPinyin(i))
    new_items=new_each_item[0]
    for i in new_each_item:
        new_items+=','+i
    for i in range(len(each_item)):
        print(each_item[i]+','+new_each_item[i])