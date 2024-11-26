class ReferenceCoordinates:
    def __init__(self, v1: list[int], v2: list[int], h1: list[int], h2: list[int]):
        self.v1 = v1
        self.v2 = v2
        self.h1 = h1
        self.h2 = h2


referenceMapping = {
    "20221115_113319.jpg": ReferenceCoordinates([1626,1656], [1633,2047], [1282,1505],[1869,1501]),
    "20221115_113328.jpg": ReferenceCoordinates([1800,1823], [1804,2137], [1853,1731],  [2241,1720]),
    "20221115_113340.jpg": ReferenceCoordinates([1865,1573], [1880,1990], [1571,1317], [2296,1470]),
    "20221115_113346.jpg": ReferenceCoordinates([1080,1338], [1114,2138], [1003,1242], [1695,1327]),
    "20221115_113356.jpg": ReferenceCoordinates([1716,1634], [1723,1996], [1514,1593], [2037,1540]),
    "20221115_113401.jpg": ReferenceCoordinates([1632,1172], [1648,1673], [1340,1166], [2044,1127]),
    "20221115_113412.jpg": ReferenceCoordinates([1479,1486], [1503,1825], [1328,1452], [1813,1314]),
    "20221115_113424.jpg": ReferenceCoordinates([1256,1097], [1297,1888], [977,1139], [1805,794]),
    "20221115_113437.jpg": ReferenceCoordinates([1064,1809], [1221,4017], [870,1241], [1794,1516]),
    "20221115_113440.jpg": ReferenceCoordinates([1035,76], [1163,2665], [1645,22], [2368,673]),
    "20221115_113635.jpg": ReferenceCoordinates([1476,1895], [1485,2213], [1092,1639], [1563,1637])
}
