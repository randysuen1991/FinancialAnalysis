import pandas as pd
import sortedcontainers as sc


class Order:
    def __init__(self, price, size, order_id=None):
        self.price = price
        self.size = size
        self.order_id = order_id


class Limit:
    def __init__(self, limit_price, size, book_parent):
        self.limit_price = limit_price
        self.size = size
        self.total_volume = 0.0
        self.parent = book_parent
        # The trade execution should be done in the order of this queue.
        self.order_queue = []

    def __repr__(self):
        return str((self.limit_price, self.size, self.total_volume))

    def on_add_order(self, price, size, order_id):
        self.order_queue.append(Order(price, size, order_id))
        self.total_volume += size

    def on_cancel_order(self, order_id):
        found = False
        for order in self.order_queue:
            if order_id == order.order_id:
                self.order_queue.remove(order)
                found = True
                break
        if not found:
            raise ValueError('There is no such order id:' + order_id)

    def on_trade(self, size):
        self.total_volume += size
        for order in self.order_queue:
            if size > order.size:
                self.order_queue.remove(order)
                size -= order.size
            elif size == order.size:
                self.order_queue.remove(order)
                break
            else:
                order.size -= size
                break


class Book:
    def __init__(self, instrument, verbose=False):
        self.instrument = instrument
        # small to large
        self.buy_limits = sc.SortedDict()
        self.sell_limits = sc.SortedDict()
        self.highest_buy = None
        self.lowest_sell = None

        self.total_buy_volume = 0
        self.total_sell_volume = 0

        self.buy_count = 0
        self.sell_count = 0

        self.verbose = verbose

    # The argument limits should be a dictionary with keys, 'ask' and 'bid'.
    # The value should be a list with elements being tuples (price, size).
    @classmethod
    def Construct_with_Limits(cls, instrument,  limits):
        book = cls(instrument)
        ask_limits = limits['ask']
        bid_limits = limits['bid']
        for ask in ask_limits:
            order = Order(ask[0], ask[1])
            limit = Limit(ask[0], ask[1], book)
            limit.order_queue.append(order)
            book.sell_limits[ask[0]] = limit

        for bid in bid_limits:
            order = Order(bid[0], bid[1])
            limit = Limit(bid[0], bid[1], book)
            limit.order_queue.append(order)
            book.buy_limits[bid[0]] = limit

        return book

    # Unfinished
    def on_trade(self, side, price, size):
        if type(size) == str:
            size = eval(size)
        if side == 'buy':
            self.buy_count += 1
            self.total_buy_volume += size
            if self.verbose:
                print(self.instrument, ' buy:', self.total_buy_volume, ' count:', self.buy_count)
        elif side == 'sell':
            self.sell_count += 1
            self.total_sell_volume += size
            if self.verbose:
                print(self.instrument, ' sell:', self.total_sell_volume, ' count:', self.sell_count)
        else:
            raise ValueError('on trade: does not recognize size=' + side)


    def on_level_update(self, side, price, size, order_id=None):
        price = eval(price)
        size = eval(size)
        if side == 'buy':
            if size == 0:
                self.buy_limits.pop(price)
            else:
                if price in self.buy_limits.keys():
                    self.buy_limits[price].on_add_order(price, size, order_id)
                    self.buy_limits[price].size += size
                else:
                    limit = Limit(price, size, self.buy_limits, self)
                    limit.on_add_order(price, size, order_id)
                    self.buy_limits[price] = limit

            # It's a Limit object.
            self.highest_buy = self.buy_limits.peekitem(-1)[1]

        elif side == 'sell':
            if size == 0:
                self.sell_limits.pop(price)
            else:
                if price in self.sell_limits.keys():
                    self.sell_limits[price].on_add_order(price, size, order_id)
                    self.sell_limits[price].size += size
                else:
                    limit = Limit(price, size, self.sell_limits, self)
                    limit.on_add_order(price, size, order_id)
                    self.sell_limits[price] = limit

            self.lowest_sell = self.sell_limits.peekitem(0)[1]
        else:
            raise ValueError('does not recognize side == ' + side)

    def on_order_update(self, side, price, new_size, order_id):
        if side == 'buy':
            limit = self.buy_limits[price]
        else:
            limit = self.sell_limits[price]

        orders = limit.order_queue

        for order in orders:
            if order.order_id == order_id:
                old_size = order.size
                order.size = new_size
                limit.size -= old_size
                limit.size += new_size
                break
