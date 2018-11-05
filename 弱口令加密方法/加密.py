def comparePersonData(data1, data2):
    diff = 0
    # for v1, v2 in data1, data2:
        # diff += (v1 - v2)**2
    for i in xrange(len(data1)):
        diff += (data1[i] - data2[i])**2
    diff = numpy.sqrt(diff)
    #print diff
    if(diff < 0.6):
        str3 = bytes(data2)
        h = hmac.new(str3)
        #h_str = h.hexdigest()
        hc = hmac.new(b"key")
        hc.update(str3)
        #hash_bytes = hc.digest()
        hash_bytes = hmac.new(b"key")
        hash_str = hash_bytes.hexdigest()
        # print(hash_str)
        base64_str = base64.urlsafe_b64encode(hash_str)
        #print("It's the same person")
        return base64_str
    else:
        print "It's not the same person"

